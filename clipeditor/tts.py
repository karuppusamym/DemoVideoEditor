"""Offline text-to-speech.

    voices() -> [{"id", "name", "engine", "language"}]   # id: "sapi:<name>" | "piper:<model path>"
    synthesize(text, voice_id=None, rate=0, out_path=None, volume=100) -> wav path
        rate: -10..10 (SAPI semantics; mapped to length_scale for Piper)
    engines() -> {"sapi": "ok" | reason, "piper": "ok" | reason}

Primary engine: Windows SAPI via PowerShell System.Speech (text passed through a UTF-8 temp file,
never on the command line). Optional: Piper, used only if the `piper` package is importable;
its .onnx voices are looked up in %APPDATA%/ClipEditor/tts/piper_voices.
Output WAVs go to %APPDATA%/ClipEditor/tts. Blocking: call from a worker thread.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import time
import uuid

from .audio_fx import app_dir

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_LIST_PS = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
  @{ name = $_.VoiceInfo.Name; language = $_.VoiceInfo.Culture.Name; gender = [string]$_.VoiceInfo.Gender } })
$s.Dispose()
ConvertTo-Json -InputObject $v -Compress
"""

_SPEAK_PS = r"""
param([string]$TextFile, [string]$OutFile, [string]$Voice = '', [int]$Rate = 0, [int]$Volume = 100)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  if ($Voice -ne '') { $s.SelectVoice($Voice) }
  $s.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate))
  $s.Volume = [Math]::Max(0, [Math]::Min(100, $Volume))
  $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  $s.SetOutputToWaveFile($OutFile, $fmt)
  $s.Speak($text)
} finally { $s.Dispose() }
"""

_sapi_cache: list | None = None
_sapi_error: str | None = None


def _powershell(script: str, args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    fd, ps1 = tempfile.mkstemp(suffix=".ps1", prefix="clipeditor_tts_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
            f.write(script)
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-File", ps1, *args]
        return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
    finally:
        try:
            os.remove(ps1)
        except OSError:
            pass


def _sapi_voices() -> list[dict]:
    global _sapi_cache, _sapi_error
    if _sapi_cache is not None:
        return _sapi_cache
    if os.name != "nt":
        _sapi_error, _sapi_cache = "Windows only", []
        return []
    try:
        p = _powershell(_LIST_PS, [], timeout=60)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", "replace").strip() or "PowerShell failed")
        data = json.loads(p.stdout.decode("utf-8-sig", "replace").strip() or "[]")
        if isinstance(data, dict):
            data = [data]
        _sapi_cache = [{"id": "sapi:" + v["name"], "name": v["name"], "engine": "sapi",
                        "language": v.get("language", ""), "gender": v.get("gender", "")} for v in data]
        _sapi_error = None if _sapi_cache else "no SAPI voices installed"
    except Exception as e:  # noqa: BLE001
        _sapi_error, _sapi_cache = str(e), []
    return _sapi_cache


def _piper_module():
    try:
        import piper  # noqa: F401
        return piper
    except Exception:
        return None


def piper_voice_dir() -> str:
    return app_dir("tts", "piper_voices")


def _piper_voices() -> list[dict]:
    if _piper_module() is None:
        return []
    out = []
    for model in sorted(glob.glob(os.path.join(piper_voice_dir(), "*.onnx"))):
        name = os.path.splitext(os.path.basename(model))[0]
        lang = ""
        try:
            with open(model + ".json", encoding="utf-8") as f:
                cfg = json.load(f)
            lang = (cfg.get("language") or {}).get("code", "") or cfg.get("espeak", {}).get("voice", "")
        except Exception:
            pass
        out.append({"id": "piper:" + model, "name": name, "engine": "piper", "language": lang})
    return out


def engines() -> dict:
    _sapi_voices()
    if _piper_module() is None:
        piper = "unavailable (piper-tts not installed)"
    elif not _piper_voices():
        piper = f"no voices (put .onnx + .onnx.json models in {piper_voice_dir()})"
    else:
        piper = "ok"
    return {"sapi": "ok" if _sapi_cache else (_sapi_error or "unavailable"), "piper": piper}


def voices() -> list[dict]:
    return _sapi_voices() + _piper_voices()


def _default_out() -> str:
    return os.path.join(app_dir("tts"), time.strftime("tts_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6] + ".wav")


def synthesize(text: str, voice_id: str | None = None, rate: int = 0, out_path: str | None = None,
               volume: int = 100) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to say: the text is empty.")
    out_path = os.path.abspath(out_path or _default_out())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if voice_id and voice_id.startswith("piper:"):
        return _piper_synth(text, voice_id[6:], rate, out_path)
    voice = voice_id[5:] if voice_id and voice_id.startswith("sapi:") else (voice_id or "")
    if os.name != "nt":
        raise RuntimeError("Windows speech (SAPI) is only available on Windows.")
    fd, txt = tempfile.mkstemp(suffix=".txt", prefix="clipeditor_tts_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        p = _powershell(_SPEAK_PS, ["-TextFile", txt, "-OutFile", out_path, "-Voice", voice,
                                    "-Rate", str(int(rate)), "-Volume", str(int(volume))],
                        timeout=60 + len(text) / 5)
    finally:
        try:
            os.remove(txt)
        except OSError:
            pass
    if p.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) <= 44:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("Text-to-speech failed: " + (err[0] if err else f"exit {p.returncode}"))
    return out_path


def _piper_synth(text: str, model: str, rate: int, out_path: str) -> str:
    mod = _piper_module()
    if mod is None:
        raise RuntimeError("Piper is not installed.")
    import wave
    PiperVoice = getattr(mod, "PiperVoice", None)
    if PiperVoice is None:
        from piper.voice import PiperVoice  # type: ignore
    voice = PiperVoice.load(model)
    length_scale = max(0.5, min(2.0, 1.0 - 0.05 * int(rate)))  # rate>0 -> faster
    with wave.open(out_path, "wb") as wf:
        if hasattr(voice, "synthesize_wav"):  # piper-tts >= 1.3
            try:
                from piper import SynthesisConfig  # type: ignore
                voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=length_scale))
            except ImportError:
                voice.synthesize_wav(text, wf)
        else:  # piper-tts 1.2
            voice.synthesize(text, wf, length_scale=length_scale)
    return out_path
