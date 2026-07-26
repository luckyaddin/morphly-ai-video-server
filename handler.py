import logging
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional

import runpod
from supabase import Client, create_client

from ltx_engine import LTXEngine
from generation_timing import (
    build_generation_timing,
    duration_differs_from_request,
    probe_video_duration,
)


MODEL_ROOT = Path(
    os.getenv(
        "MODEL_ROOT",
        "/runpod-volume/models",
    )
)

CHECKPOINT_PATH = os.getenv(
    "LTX_CHECKPOINT_PATH",
    str(
        MODEL_ROOT
        / "ltx-2.3"
        / "ltx-2.3-22b-dev.safetensors"
    ),
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
    str(
        MODEL_ROOT
        / "gemma-3-12b"
    ),
)

IC_LORA_PATH = os.getenv(
    "LTX_IC_LORA_PATH"
)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY",
    "",
).strip()

SUPABASE_BUCKET = os.getenv(
    "SUPABASE_BUCKET",
    "morphly-generated-videos",
).strip()

SIGNED_URL_EXPIRES_IN = int(
    os.getenv(
        "SIGNED_URL_EXPIRES_IN",
        "86400",
    )
)


engine = LTXEngine()
_supabase_client: Optional[Client] = None
logger = logging.getLogger("morphly_worker")


def _require_file(
    path: str,
    label: str,
) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def _require_directory(
    path: str,
    label: str,
) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def _get_supabase_client() -> Client:
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured."
        )

    if not SUPABASE_SECRET_KEY:
        raise RuntimeError(
            "SUPABASE_SECRET_KEY is not configured."
        )

    if not SUPABASE_BUCKET:
        raise RuntimeError(
            "SUPABASE_BUCKET is not configured."
        )

    _supabase_client = create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY,
    )

    return _supabase_client


def _extract_signed_url(
    response: Any,
) -> str:
    if isinstance(response, dict):
        signed_url = (
            response.get("signedURL")
            or response.get("signed_url")
        )

        if signed_url:
            return str(signed_url)

    signed_url = getattr(
        response,
        "signed_url",
        None,
    )

    if signed_url:
        return str(signed_url)

    raise RuntimeError(
        "Supabase did not return a signed URL."
    )


def _upload_video(
    output_path: str,
    job_id: str,
) -> dict[str, Any]:
    local_file = Path(output_path)

    if not local_file.is_file():
        raise FileNotFoundError(
            f"Generated video not found: {output_path}"
        )

    client = _get_supabase_client()

    storage_path = (
        f"generated/{job_id}.mp4"
    )

    with local_file.open("rb") as video_file:
        client.storage.from_(
            SUPABASE_BUCKET
        ).upload(
            path=storage_path,
            file=video_file,
            file_options={
                "content-type": "video/mp4",
                "cache-control": "3600",
                "upsert": "false",
            },
        )

    signed_response = (
        client.storage
        .from_(SUPABASE_BUCKET)
        .create_signed_url(
            storage_path,
            SIGNED_URL_EXPIRES_IN,
        )
    )

    signed_url = _extract_signed_url(
        signed_response
    )

    return {
        "storage_bucket": SUPABASE_BUCKET,
        "storage_path": storage_path,
        "download_url": signed_url,
        "url_expires_in": (
            SIGNED_URL_EXPIRES_IN
        ),
        "file_size_bytes": (
            local_file.stat().st_size
        ),
    }


def load_engine() -> None:
    _require_file(
        CHECKPOINT_PATH,
        "LTX checkpoint",
    )

    _require_file(
        DISTILLED_LORA_PATH,
        "Distilled LoRA",
    )

    _require_file(
        SPATIAL_UPSAMPLER_PATH,
        "Spatial upsampler",
    )

    _require_directory(
        GEMMA_ROOT,
        "Gemma model directory",
    )

    if IC_LORA_PATH:
        _require_file(
            IC_LORA_PATH,
            "IC-LoRA",
        )

        _require_file(
            DISTILLED_CHECKPOINT_PATH,
            "LTX distilled checkpoint",
        )

    engine.load_models(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora_path=(
            DISTILLED_LORA_PATH
        ),
        spatial_upsampler_path=(
            SPATIAL_UPSAMPLER_PATH
        ),
        gemma_root=GEMMA_ROOT,
        ic_lora_path=IC_LORA_PATH,
        distilled_checkpoint_path=(
            DISTILLED_CHECKPOINT_PATH
        ),
        device="cuda",
    )


def _validated_integer(
    data: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(
        data.get(
            key,
            default,
        )
    )

    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between "
            f"{minimum} and {maximum}."
        )

    return value


def _validated_float(
    data: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = float(
        data.get(
            key,
            default,
        )
    )

    if value < minimum or value > maximum:
        raise ValueError(
            f"{key} must be between "
            f"{minimum} and {maximum}."
        )

    return value


def _required_integer(
    data: dict[str, Any],
    key: str,
    minimum: int,
    maximum: int,
) -> int:
    if key not in data or data[key] is None:
        raise ValueError(
            f"{key} is required."
        )

    value = data[key]

    if isinstance(value, bool):
        raise ValueError(
            f"{key} must be an integer."
        )

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key} must be an integer."
        ) from exc

    if float(value) != integer:
        raise ValueError(
            f"{key} must be an integer."
        )

    if integer < minimum or integer > maximum:
        raise ValueError(
            f"{key} must be between "
            f"{minimum} and {maximum}."
        )

    return integer


def handler(
    job: dict[str, Any],
) -> dict[str, Any]:
    job_input = job.get(
        "input"
    ) or {}

    output_path: Optional[str] = None

    try:
        mode = str(
            job_input.get(
                "mode",
                "",
            )
        ).strip()

        if mode == "health":
            health = engine.health_check()

            health["supabase_configured"] = bool(
                SUPABASE_URL
                and SUPABASE_SECRET_KEY
                and SUPABASE_BUCKET
            )

            health["storage_bucket"] = (
                SUPABASE_BUCKET
            )

            return health

        if mode not in {
            "text_to_video",
            "image_to_video",
            "video_to_video",
        }:
            raise ValueError(
                "mode must be text_to_video, "
                "image_to_video, or video_to_video."
            )

        if not engine.is_loaded:
            load_engine()

        prompt = str(
            job_input.get(
                "prompt",
                "",
            )
        ).strip()

        if not prompt:
            raise ValueError(
                "prompt is required."
            )

        width = _validated_integer(
            job_input,
            "width",
            768,
            256,
            1536,
        )

        height = _validated_integer(
            job_input,
            "height",
            512,
            256,
            1536,
        )

        if width % 64 != 0:
            raise ValueError(
                "width must be divisible by 64."
            )

        if height % 64 != 0:
            raise ValueError(
                "height must be divisible by 64."
            )

        requested_duration_seconds = (
            _required_integer(
                job_input,
                "requested_duration_seconds",
                1,
                60,
            )
        )

        frames = _required_integer(
            job_input,
            "frames",
            9,
            1201,
        )

        fps = _required_integer(
            job_input,
            "fps",
            1,
            60,
        )

        timing = build_generation_timing(
            requested_duration_seconds=(
                requested_duration_seconds
            ),
            frames=frames,
            fps=fps,
        )

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
            job_input.get(
                "negative_prompt",
                "",
            )
        ).strip()

        output_dir = (
            Path(tempfile.gettempdir())
            / "morphly-output"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        job_id = str(
            job_input.get(
                "job_id",
                job.get(
                    "id",
                    "local-job",
                ),
            )
        )

        output_path = str(
            output_dir
            / f"{job_id}.mp4"
        )

        if mode == "text_to_video":
            result = (
                engine.generate_text_to_video(
                    prompt=prompt,
                    negative_prompt=(
                        negative_prompt
                    ),
                    width=width,
                    height=height,
                    timing=timing,
                    seed=seed,
                    guidance_scale=(
                        guidance_scale
                    ),
                    inference_steps=(
                        inference_steps
                    ),
                    output_path=output_path,
                )
            )

        elif mode == "image_to_video":
            image_path = str(
                job_input.get(
                    "image_path",
                    "",
                )
            ).strip()

            if not image_path:
                raise ValueError(
                    "image_path is required."
                )

            result = (
                engine.generate_image_to_video(
                    image_path=image_path,
                    prompt=prompt,
                    negative_prompt=(
                        negative_prompt
                    ),
                    width=width,
                    height=height,
                    timing=timing,
                    seed=seed,
                    guidance_scale=(
                        guidance_scale
                    ),
                    inference_steps=(
                        inference_steps
                    ),
                    output_path=output_path,
                )
            )

        elif mode == "video_to_video":
            video_path = str(
                job_input.get(
                    "video_path",
                    "",
                )
            ).strip()

            if not video_path:
                raise ValueError(
                    "video_path is required."
                )

            result = (
                engine.generate_video_to_video(
                    video_path=video_path,
                    prompt=prompt,
                    width=width,
                    height=height,
                    timing=timing,
                    seed=seed,
                    output_path=output_path,
                )
            )

        else:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        if not Path(output_path).is_file():
            raise RuntimeError(
                "Generation finished without "
                "creating an MP4."
            )

        actual_duration_seconds = (
            probe_video_duration(
                output_path,
            )
        )

        duration_mismatch = (
            duration_differs_from_request(
                requested_duration_seconds=(
                    timing.requested_duration_seconds
                ),
                actual_duration_seconds=(
                    actual_duration_seconds
                ),
            )
        )

        if duration_mismatch:
            logger.warning(
                "Generated MP4 duration mismatch for %s: "
                "requested=%ss actual=%.3fs frames=%s fps=%s",
                job_id,
                timing.requested_duration_seconds,
                actual_duration_seconds,
                timing.frames,
                timing.fps,
            )

        upload_result = _upload_video(
            output_path=output_path,
            job_id=job_id,
        )

        result.pop(
            "output_path",
            None,
        )

        result.update(
            upload_result
        )

        result.update(
            timing.output_metadata(
                actual_duration_seconds
            )
        )

        result["duration_mismatch"] = (
            duration_mismatch
        )

        return result

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": (
                type(exc).__name__
            ),
            "traceback": (
                traceback.format_exc()
            ),
        }

    finally:
        if output_path:
            try:
                Path(output_path).unlink(
                    missing_ok=True
                )
            except Exception:
                pass


runpod.serverless.start(
    {
        "handler": handler,
    }
)
