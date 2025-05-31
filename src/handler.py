from http.client import HTTPException
import io
import os
import base64
import re
import uuid
from typing import Dict
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch
import runpod
import trimesh

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.pipelines import export_to_trimesh

# 環境設定
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HOME"] = "/workspace"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/transformers"
os.environ["XDG_CACHE_HOME"] = "/workspace"
os.environ["HY3DGEN_MODELS"] = "/workspace/hy3dgen"
os.environ["TMPDIR"] = "/workspace/tmp"

device = "cuda"

# モデル初期化
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2",
    subfolder='hunyuan3d-dit-v2-0-turbo',
    use_safetensors=True,
    device=device,
)

# 許容する画像フォーマット(pillowでの表記)
ALLOWED_IMAGE_FORMATS = ["JPEG", "PNG"]
MAX_IMAGE_DIMENSION = 5000

def create_error_response(error_message: str, details: str = "") -> Dict:
    return {
        "status": "error",
        "error": error_message,
        "logs": {
            "message": details if details else error_message
        }
    }

def handler(job: Dict[str, any]) -> Dict:
    try:
        job_input = job.get('input', {})
        image_data_url = job_input.get('image', '')
        parameters = job_input.get('parameters', {})

        if not image_data_url:
            return create_error_response(
                "Input 'image' (data URL) is required.",
                "The 'image' field in the input is missing or empty."
            )

        match = re.fullmatch(r"data:(image/[\w.+-]+);base64,(.+)", image_data_url)
        if not match:
            if not image_data_url.startswith("data:image"):
                 return create_error_response(
                    "Invalid data URL prefix for 'image'.",
                    "The 'image' field must start with 'data:image/'."
                )
            pass
        declared_mime_type = match.group(1) if match else "unknown"
        encoded_data = match.group(2) if match else image_data_url.split(',', 1)[-1]

        try:
            image_bytes = base64.b64decode(encoded_data)
        except base64.binascii.Error as e:
            return create_error_response(
                "Invalid base64 encoding for image data.",
                f"Failed to decode base64 string: {str(e)}"
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError:
            return create_error_response(
                "Cannot identify image file.",
                "The provided image data could not be opened or identified as a valid image."
            )
        except Exception as e:
            return create_error_response(
                "Error opening image.",
                f"An unexpected error occurred while opening the image: {str(e)}"
            )

        actual_image_format = image.format
        if actual_image_format not in ALLOWED_IMAGE_FORMATS:
            return create_error_response(
                "Unsupported image format.",
                f"Only JPEG and PNG formats are supported. Detected format: {actual_image_format}"
            )

        try:
            image = image.convert("RGBA")
        except Exception as e:
            return create_error_response(
                "Error converting image to RGBA.",
                f"Failed to convert image to RGBA format: {str(e)}"
            )

        # 画像サイズのチェック
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            return create_error_response(
                "Image dimensions exceed maximum limit.",
                f"Image dimensions ({image.width}x{image.height}) exceed the maximum allowed limit of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels."
            )

        # 入力パラメータの取得とデフォルト値の設定
        num_inference_steps = parameters.get('num_inference_steps', 30)
        guidance_scale = parameters.get('guidance_scale', 7.5)
        octree_resolution = parameters.get('octree_resolution', 256)
        num_chunks = parameters.get('num_chunks', 200000)
        seed = parameters.get('seed', 1234)

        # 推論処理
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )

        mesh_obj = export_to_trimesh(outputs)[0]

        # glb形式でエクスポート。include_normals=Falseじゃないと落ちるかも？
        export_kwargs = {'file_type': 'glb', 'include_normals': False}
        with io.BytesIO() as f:
            mesh_obj.export(f, **export_kwargs)
            f.seek(0)
            encoded_mesh = base64.b64encode(f.read()).decode('utf-8')

        used_parameters = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution,
            "num_chunks": num_chunks,
            "seed": seed
        }

        return {
            "id": str(uuid.uuid4()),
            "status": "success",
            "input_summary": {
                "declared_image_mime_type": declared_mime_type,
                "actual_image_format": actual_image_format,
                "image_dimensions": {"width": image.width, "height": image.height},
                "parameters_used": used_parameters
            },
            "model_output": {
                "filename": "model.glb",
                "mime_type": "model/gltf-binary",
                "data": f"data:model/gltf-binary;base64,{encoded_mesh}"
            },
            "logs": {
                "message": "3D model generated successfully",
            }
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "logs": {
                "message": "An unexpected error occurred during model inference."
            }
        }


runpod.serverless.start({"handler": handler})