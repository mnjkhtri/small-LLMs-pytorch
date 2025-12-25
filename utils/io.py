import os
import requests
from pathlib import Path
from tqdm import tqdm
from utils.dump import LOGGER


def download_safetensors(url, save_directory):
    LOGGER.block("weights: download/cache")
    cache_path = Path(".cache") / save_directory
    cache_path.mkdir(parents=True, exist_ok=True)
    destination_path = cache_path / "model.safetensors"
    hf_token = os.getenv("HF_TOKEN")
    LOGGER.step(f"📦 cache={cache_path} dst={destination_path}")
    LOGGER.step(f"🌐 src={url} token={'set' if hf_token else 'not set'}")
    if destination_path.exists():
        LOGGER.step("🗄️ cache hit")
        LOGGER.done("weights ready")
        return destination_path
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    with LOGGER.timer() as t:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            LOGGER.step(f"📏 size={total/1e9:.3f}GB" if total else "📏 size=unknown")
            with open(destination_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="weights") as p:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk); p.update(len(chunk))
    LOGGER.done(f"downloaded -> {destination_path}", elapsed=t.dt)
    return destination_path