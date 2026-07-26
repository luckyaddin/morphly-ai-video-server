import os

# Must be configured before importing torch.
# Helps reduce CUDA memory fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import sys
import logging
import threading
from typing import Any, Dict, Optional

import torch


# Add LTX packages to sys.path.
LTX_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ltx",
)

sys.path.insert(
    0,
    os.path.join(
        LTX_ROOT,
        "packages",
        "ltx-core",
        "src",
    ),
)

sys.path.insert(
    0,
    os.path.join(
        LTX_ROOT,
        "packages",
        "ltx-pipelines",
        "src",
    ),
)


try:
    from ltx_pipelines.ti2vid_two_stages import (
        TI2VidTwoStagesPipeline,
    )
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.types import OffloadMode

    from ltx_core.model.video_vae import (
        TilingConfig,
        get_video_chunks_number,
    )
    from ltx_core.loader import (
        LoraPathStrengthAndSDOps,
        LTXV_LORA_COMFY_RENAMING_MAP,
    )
    from ltx_core.components.guiders import (
        MultiModalGuiderParams,
    )

except ImportError as exc:
    raise ImportError(
        "Failed to import from ltx-pipelines or ltx-core. "
        "Make sure the LTX submodule is correctly initialized. "
        f"Error: {exc}"
    ) from exc


logger = logging.getLogger("ltx_engine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class LTXEngine:
    def __init__(self) -> None:
        self.ti2vid_pipeline = None
        self.ic_lora_pipeline = None
        self.lock = threading.Lock()
        self.is_loaded = False

        # CPU offloading prevents the LTX model and Gemma model from
        # occupying GPU VRAM simultaneously.
        self.offload_mode = self._resolve_offload_mode()

    @staticmethod
    def _resolve_offload_mode() -> OffloadMode:
        """
        Reads the offload mode from the environment.

        Supported values:
        - cpu: recommended for the current RunPod worker
        - disk: lowest memory usage but slower
        - none: fastest but requires significantly more available VRAM
        """
        value = os.getenv(
            "LTX_OFFLOAD_MODE",
            "cpu",
        ).strip().lower()

        try:
            return OffloadMode(value)
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in OffloadMode)
            raise ValueError(
                f"Invalid LTX_OFFLOAD_MODE '{value}'. "
                f"Supported values: {allowed}"
            ) from exc

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_models(
        self,
        checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        distilled_lora_path: Optional[str] = None,
        ic_lora_path: Optional[str] = None,
        distilled_checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        """
        Loads the LTX models once and reuses them for subsequent requests.
        """
        with self.lock:
            if self.is_loaded:
                logger.info("Models are already loaded.")
                return

            logger.info(
                "Loading LTX models with offload mode: %s",
                self.offload_mode.value,
            )

            try:
                self._clear_cuda_cache()

                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA was requested but is not available."
                    )

                device_obj = torch.device(device)

                if not distilled_lora_path:
                    raise ValueError(
                        "distilled_lora_path is required for "
                        "TI2VidTwoStagesPipeline."
                    )

                distilled_lora = [
                    LoraPathStrengthAndSDOps(
                        path=distilled_lora_path,
                        strength=1.0,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                ]

                logger.info(
                    "Initializing TI2VidTwoStagesPipeline..."
                )

                self.ti2vid_pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=checkpoint_path,
                    distilled_lora=distilled_lora,
                    spatial_upsampler_path=spatial_upsampler_path,
                    gemma_root=gemma_root,
                    loras=[],
                    device=device_obj,
                    offload_mode=self.offload_mode,
                )

                if ic_lora_path:
                    if not distilled_checkpoint_path:
                        raise ValueError(
                            "distilled_checkpoint_path is required "
                            "when ic_lora_path is provided."
                        )

                    logger.info(
                        "Initializing ICLoraPipeline..."
                    )

                    ic_lora = [
                        LoraPathStrengthAndSDOps(
                            path=ic_lora_path,
                            strength=1.0,
                            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                        )
                    ]

                    self.ic_lora_pipeline = ICLoraPipeline(
                        distilled_checkpoint_path=(
                            distilled_checkpoint_path
                        ),
                        spatial_upsampler_path=(
                            spatial_upsampler_path
                        ),
                        gemma_root=gemma_root,
                        loras=ic_lora,
                        device=device_obj,
                        offload_mode=self.offload_mode,
                    )

                else:
                    logger.info(
                        "No IC LoRA path provided; "
                        "skipping ICLoraPipeline initialization."
                    )

                self.is_loaded = True
                self._clear_cuda_cache()

                logger.info(
                    "LTX models loaded successfully using %s offloading.",
                    self.offload_mode.value,
                )

            except Exception as exc:
                self.ti2vid_pipeline = None
                self.ic_lora_pipeline = None
                self.is_loaded = False
                self._clear_cuda_cache()

                logger.exception(
                    "Failed to load LTX models."
                )

                raise RuntimeError(
                    f"Error loading models: {exc}"
                ) from exc

    @torch.inference_mode()
    def warmup(self) -> None:
        """
        Performs a small warmup generation.
        """
        logger.info("Warming up LTX engine...")

        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError(
                    "Models must be loaded before warmup."
                )

            try:
                self._clear_cuda_cache()

                tiling_config = TilingConfig.default()

                self.ti2vid_pipeline(
                    prompt="A black screen",
                    negative_prompt="",
                    seed=42,
                    height=256,
                    width=256,
                    num_frames=9,
                    frame_rate=8.0,
                    num_inference_steps=1,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=1.0
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=1.0
                    ),
                    images=[],
                    tiling_config=tiling_config,
                )

                self._clear_cuda_cache()
                logger.info("Warmup complete.")

            except Exception as exc:
                self._clear_cuda_cache()

                logger.exception(
                    "Warmup failed."
                )

                raise RuntimeError(
                    f"Warmup failed: {exc}"
                ) from exc

    @torch.inference_mode()
    def generate_text_to_video(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        fps: float,
        frames: int,
        seed: int,
        guidance_scale: float,
        inference_steps: int,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Generates a video from a text prompt.
        """
        logger.info(
            "Received text-to-video request: '%s'",
            prompt,
        )

        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError(
                    "TI2Vid pipeline is not loaded."
                )

            try:
                self._clear_cuda_cache()

                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(
                    frames,
                    tiling_config,
                )

                video, audio = self.ti2vid_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frames,
                    frame_rate=fps,
                    num_inference_steps=inference_steps,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=guidance_scale
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=guidance_scale
                    ),
                    images=[],
                    tiling_config=tiling_config,
                )

                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

                self._clear_cuda_cache()

                logger.info(
                    "Text-to-video generation successful."
                )

                return {
                    "success": True,
                    "output_path": output_path,
                    "fps": fps,
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "seed": seed,
                }

            except Exception as exc:
                self._clear_cuda_cache()

                logger.exception(
                    "Error during text-to-video generation."
                )

                raise RuntimeError(
                    f"Generation failed: {exc}"
                ) from exc

    @torch.inference_mode()
    def generate_image_to_video(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        fps: float,
        frames: int,
        seed: int,
        guidance_scale: float,
        inference_steps: int,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Generates a video from an image and text prompt.
        """
        logger.info(
            "Received image-to-video request: '%s' with image: %s",
            prompt,
            image_path,
        )

        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError(
                    "TI2Vid pipeline is not loaded."
                )

            try:
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Input image not found: {image_path}"
                    )

                self._clear_cuda_cache()

                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(
                    frames,
                    tiling_config,
                )

                images = [
                    ImageConditioningInput(
                        path=image_path,
                        frame_idx=0,
                        strength=1.0,
                    )
                ]

                video, audio = self.ti2vid_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frames,
                    frame_rate=fps,
                    num_inference_steps=inference_steps,
                    video_guider_params=MultiModalGuiderParams(
                        cfg_scale=guidance_scale
                    ),
                    audio_guider_params=MultiModalGuiderParams(
                        cfg_scale=guidance_scale
                    ),
                    images=images,
                    tiling_config=tiling_config,
                )

                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

                self._clear_cuda_cache()

                logger.info(
                    "Image-to-video generation successful."
                )

                return {
                    "success": True,
                    "output_path": output_path,
                    "fps": fps,
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "seed": seed,
                }

            except Exception as exc:
                self._clear_cuda_cache()

                logger.exception(
                    "Error during image-to-video generation."
                )

                raise RuntimeError(
                    f"Generation failed: {exc}"
                ) from exc

    @torch.inference_mode()
    def generate_video_to_video(
        self,
        video_path: str,
        prompt: str,
        width: int,
        height: int,
        fps: float,
        frames: int,
        seed: int,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Transforms a video using the optional IC-LoRA pipeline.
        """
        logger.info(
            "Received video-to-video request: '%s' with video: %s",
            prompt,
            video_path,
        )

        with self.lock:
            if not self.is_loaded or self.ic_lora_pipeline is None:
                raise RuntimeError(
                    "IC LoRA pipeline is not loaded. "
                    "Cannot perform video-to-video."
                )

            try:
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(
                        f"Input video not found: {video_path}"
                    )

                self._clear_cuda_cache()

                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(
                    frames,
                    tiling_config,
                )

                video_conditioning = [
                    (video_path, 1.0)
                ]

                video, audio = self.ic_lora_pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frames,
                    frame_rate=fps,
                    images=[],
                    video_conditioning=video_conditioning,
                    tiling_config=tiling_config,
                )

                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

                self._clear_cuda_cache()

                logger.info(
                    "Video-to-video generation successful."
                )

                return {
                    "success": True,
                    "output_path": output_path,
                    "fps": fps,
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "seed": seed,
                }

            except Exception as exc:
                self._clear_cuda_cache()

                logger.exception(
                    "Error during video-to-video generation."
                )

                raise RuntimeError(
                    f"Generation failed: {exc}"
                ) from exc

    def health_check(self) -> Dict[str, Any]:
        """
        Returns the current engine status.
        """
        return {
            "status": (
                "healthy"
                if self.is_loaded
                else "not_loaded"
            ),
            "ti2vid_ready": self.ti2vid_pipeline is not None,
            "ic_lora_ready": self.ic_lora_pipeline is not None,
            "offload_mode": self.offload_mode.value,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }

    def shutdown(self) -> None:
        """
        Releases the pipelines and clears CUDA memory.
        """
        with self.lock:
            logger.info(
                "Shutting down LTX engine..."
            )

            self.ti2vid_pipeline = None
            self.ic_lora_pipeline = None
            self.is_loaded = False

            self._clear_cuda_cache()

            logger.info(
                "Shutdown complete."
            )