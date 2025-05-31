# download_model.py
import os
from huggingface_hub import snapshot_download

print("Starting model download script using snapshot_download...")

hf_home = os.getenv("HF_HOME", "/workspace")
hf_hub_cache = os.getenv("HF_HUB_CACHE", os.path.join(hf_home, "huggingface"))

transformers_cache = os.getenv("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
os.makedirs(transformers_cache, exist_ok=True)
hy3dgen_models_dir = os.getenv("HY3DGEN_MODELS", os.path.join(hf_home, "hy3dgen"))
os.makedirs(hy3dgen_models_dir, exist_ok=True)
tmp_dir = os.getenv("TMPDIR", os.path.join(hf_home, "tmp"))
os.makedirs(tmp_dir, exist_ok=True)

model_name = "tencent/Hunyuan3D-2"
subfolder_name = 'hunyuan3d-dit-v2-0-turbo'

print(f"Ensuring cache directory exists: {hf_hub_cache}")
os.makedirs(hf_hub_cache, exist_ok=True)

print(f"Attempting to download model: {model_name}, specific subfolder: {subfolder_name}")
print(f"Files will be downloaded to Hugging Face cache structure within: {hf_hub_cache}")

try:
    downloaded_repo_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=[f"{subfolder_name}/*", f"{subfolder_name}/**/**"],
        cache_dir=hf_hub_cache,
        local_dir_use_symlinks=False,
    )

    print(f"Successfully initiated download for repository {model_name}.")
    print(f"Files are expected to be in cache structure. Base path: {downloaded_repo_path}")
except Exception as e:
    print(f"Error during model download with snapshot_download: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Model download script finished successfully.")