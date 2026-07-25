import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import runpod

from ltx_engine import LTXEngine


MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/runpod-volume/models"))

CHECKPOINT_PATH = os.getenv(
    "LTX_CHECKPOINT_PATH",
    str(MODEL_ROOT / "ltx-2.3" / "ltx-2.3-22b-dev.safetensors"),
)

DISTILLED_CHECKPOINT_PATH = os.getenv(
    "LTX_DISTILLED_CHECKPOINT_PATH",
    str(
        MODEL_ROOT
        / "ltx-2.3"
        / "ltx-2.3-22b-distilled-1.1.safetensors"
    ),
)

DISTILLED_LORA_PATH = os.getenv(
    "LTX_DISTILLED_LORA_PATH",
    str(
        MODEL_ROOT
        / "ltx-2.3"
        / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    ),
)

SPATIAL_UPSAMPLER_PATH = os.getenv(
    "LTX_SPATIAL_UPSAMPLER_PATH",
    str(
        MODEL_ROOT
        / "ltx-2.3"
        / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    ),
)

GEMMA_ROOT = os.getenv(
    "LTX_GEMMA_ROOT",
    str(MODEL_ROOT / "gemma-3-12b"),
)

IC_LORA_PATH = os.getenv("LTX_IC_LORA_PATH")

engine = LTXEngine()


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_directory(path: str, label: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_engine() -> None:
    _require_file(CHECKPOINT_PATH, "LTX checkpoint")
    _require_file(DISTILLED_LORA_PATH, "Distilled LoRA")
    _require_file(SPATIAL_UPSAMPLER_PATH, "Spatial upsampler")
    _require_directory(GEMMA_ROOT, "Gemma model directory")

    if IC_LORA_PATH:
        _require_file(IC_LORA_PATH, "IC-LoRA")
        _require_file(
            DISTILLED_CHECKPOINT_PATH,
            "LTX distilled checkpoint",
        )

    engine.load_models(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora_path=DISTILLED_LORA_PATH,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        ic_lora_path=IC_LORA_PATH,
        distilled_checkpoint_path=DISTILLED_CHECKPOINT_PATH,
        device="cuda",
    )


def _validated_integer(
    data: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(data.get(key, default))
    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between {minimum} and {maximum}."
        )
    return value


def _validated_float(
    data: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = float(data.get(key, default))
    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between {minimum} and {maximum}."
        )
    return value


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}

    try:
        action = str(job_input.get("action", "text_to_video"))

        if action == "health":
            return engine.health_check()

        prompt = str(job_input.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required.")

        width = _validated_integer(job_input, "width", 768, 256, 1536)
        height = _validated_integer(job_input, "height", 512, 256, 1536)

        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("width and height must be divisible by 64.")

        frames = _validated_integer(job_input, "frames", 97, 9, 257)

        if (frames - 1) % 8 != 0:
            raise ValueError("frames must follow the format 8n + 1.")

        fps = _validated_float(job_input, "fps", 24.0, 1.0, 60.0)
        seed = _validated_integer(
            job_input,
            "seed",
            42,
            0,
            2_147_483_647,
        )
        guidance_scale = _validated_float(
            job_input,
            "guidance_scale",
            3.0,
            0.0,
            20.0,
        )
        inference_steps = _validated_integer(
            job_input,
            "inference_steps",
            30,
            1,
            100,
        )

        negative_prompt = str(
            job_input.get("negative_prompt", "")
        ).strip()

        output_dir = Path(tempfile.gettempdir()) / "morphly-output"
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = str(job.get("id", "local-job"))
        output_path = str(output_dir / f"{job_id}.mp4")

        if action == "text_to_video":
            result = engine.generate_text_to_video(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                fps=fps,
                frames=frames,
                seed=seed,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                output_path=output_path,
            )

        elif action == "image_to_video":
            image_path = str(job_input.get("image_path", "")).strip()
            if not image_path:
                raise ValueError("image_path is required.")

            result = engine.generate_image_to_video(
                image_path=image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                fps=fps,
                frames=frames,
                seed=seed,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                output_path=output_path,
            )

        elif action == "video_to_video":
            video_path = str(job_input.get("video_path", "")).strip()
            if not video_path:
                raise ValueError("video_path is required.")

            result = engine.generate_video_to_video(
                video_path=video_path,
                prompt=prompt,
                width=width,
                height=height,
                fps=fps,
                frames=frames,
                seed=seed,
                output_path=output_path,
            )

        else:
            raise ValueError(f"Unsupported action: {action}")

        if not Path(output_path).is_file():
            raise RuntimeError("Generation finished without creating an MP4.")

        result["file_size_bytes"] = Path(output_path).stat().st_size

        return result

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }


load_engine()

runpod.serverless.start({"handler": handler})