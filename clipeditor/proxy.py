"""Low-resolution proxies so 4K / high-bitrate footage scrubs smoothly in the preview.

Export always uses the original files; only the preview Compositor(use_proxies=True) reads proxies.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .model import MediaAsset

PROXY_HEIGHT = 540
_pool = ThreadPoolExecutor(max_workers=1)
_NOWIN = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def proxy_dir() -> str:
    d = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "ClipEditor", "proxies")
    os.makedirs(d, exist_ok=True)
    return d


def wants_proxy(asset: MediaAsset) -> bool:
    return asset.kind == "video" and asset.has_video and (asset.height > 1080 or asset.width > 1920
                                                          or (asset.height >= 1080 and asset.fps > 40))


def proxy_path_for(asset: MediaAsset) -> str:
    st = os.stat(asset.path)
    key = hashlib.sha1(f"{asset.path}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:16]
    return os.path.join(proxy_dir(), f"{key}.mp4")


def make_proxy(asset: MediaAsset) -> str:
    from .export import ffmpeg_exe

    out = proxy_path_for(asset)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    tmp = out + ".part.mp4"
    vf = f"scale=-2:{PROXY_HEIGHT},fps=min(30\\,source_fps)" if False else f"scale=-2:{PROXY_HEIGHT}"
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error", "-i", asset.path, "-map", "0:v:0",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "15",
           "-tune", "fastdecode", "-an", tmp]
    subprocess.run(cmd, check=True, capture_output=True, creationflags=_NOWIN)
    os.replace(tmp, out)
    return out


def ensure(asset: MediaAsset, on_ready=None):
    """Create a proxy in the background if this asset needs one; sets asset.proxy_path when done."""
    if not wants_proxy(asset):
        return None
    existing = proxy_path_for(asset)
    if os.path.exists(existing):
        asset.proxy_path = existing
        return None

    def job():
        try:
            asset.proxy_path = make_proxy(asset)
            if on_ready:
                on_ready(asset.id)
        except Exception:  # noqa: BLE001 - preview just keeps using the original
            pass

    return _pool.submit(job)
