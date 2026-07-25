import sys
import os
import logging
import threading
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add ltx packages to sys.path so we can import from the ltx/ repository
LTX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ltx')
sys.path.insert(0, os.path.join(LTX_ROOT, 'packages', 'ltx-core', 'src'))
sys.path.insert(0, os.path.join(LTX_ROOT, 'packages', 'ltx-pipelines', 'src'))

try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.loader import (
    LoraPathStrengthAndSDOps,
    LTXV_LORA_COMFY_RENAMING_MAP,
)
    from ltx_core.components.guiders import MultiModalGuiderParams
except ImportError as e:
    raise ImportError(f"Failed to import from ltx-pipelines or ltx-core. Make sure the LTX submodule is correctly initialized. Error: {e}")

logger = logging.getLogger("ltx_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class LTXEngine:
    def __init__(self):
        self.ti2vid_pipeline = None
        self.ic_lora_pipeline = None
        self.lock = threading.Lock()
        self.is_loaded = False
        
    def load_models(self, 
                    checkpoint_path: str,
                    spatial_upsampler_path: str,
                    gemma_root: str,
                    distilled_lora_path: Optional[str] = None,
                    ic_lora_path: Optional[str] = None,
                    distilled_checkpoint_path: Optional[str] = None,
                    device: str = "cuda"):
        """
        Loads the LTX models into memory.
        This runs only once at startup and reuses it for every request.
        """
        with self.lock:
            if self.is_loaded:
                logger.info("Models are already loaded.")
                return

            logger.info("Loading LTX models...")
            try:
                device_obj = torch.device(device)
                
                # Setup distilled lora for TI2Vid pipeline
                if not distilled_lora_path:
                    raise ValueError(
                        "distilled_lora_path is required for TI2VidTwoStagesPipeline."
                    )
                distilled_lora = [
                    LoraPathStrengthAndSDOps(
                        path=distilled_lora_path,
                        strength=1.0,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                ]
                
                logger.info("Initializing TI2VidTwoStagesPipeline...")
                self.ti2vid_pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=checkpoint_path,
                    distilled_lora=distilled_lora,
                    spatial_upsampler_path=spatial_upsampler_path,
                    gemma_root=gemma_root,
                    loras=(),
                    device=device_obj
                )
                
                # Setup IC Lora pipeline if provided (for video-to-video)
                if ic_lora_path:
                    if not distilled_checkpoint_path:
                        raise ValueError(
                            "distilled_checkpoint_path is required when ic_lora_path is provided."
                        )
                    logger.info("Initializing ICLoraPipeline...")
                    ic_lora = [LoraPathStrengthAndSDOps(
                        ic_lora_path,
                        1.0,
                        LTXV_LORA_COMFY_RENAMING_MAP,
                    )]
                    self.ic_lora_pipeline = ICLoraPipeline(
                        distilled_checkpoint_path=distilled_checkpoint_path,
                        spatial_upsampler_path=spatial_upsampler_path,
                        gemma_root=gemma_root,
                        loras=ic_lora,
                        device=device_obj
                    )
                else:
                    logger.info("No IC LoRA path provided; skipping ICLoraPipeline initialization.")
                
                self.is_loaded = True
                logger.info("LTX models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load LTX models: {e}")
                raise RuntimeError(f"Error loading models: {e}") from e

    def warmup(self):
        """
        Performs a warmup inference to allocate memory and compile necessary components.
        """
        logger.info("Warming up LTX engine...")
        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError("Models must be loaded before warmup.")
            
            try:
                logger.info("Running warmup generation...")
                # A simple text-to-video warmup with minimal frames to initialize CUDA graphs/compilation
                tiling_config = TilingConfig.default()
                self.ti2vid_pipeline(
                    prompt="A black screen",
                    negative_prompt="",
                    seed=42,
                    height=512,
                    width=512,
                    num_frames=9,  # Minimal frames (8*k + 1)
                    frame_rate=24.0,
                    num_inference_steps=2,
                    video_guider_params=MultiModalGuiderParams(cfg_scale=3.0),
                    audio_guider_params=MultiModalGuiderParams(cfg_scale=3.0),
                    images=[],
                    tiling_config=tiling_config,
                )
                logger.info("Warmup complete.")
            except Exception as e:
                logger.error(f"Warmup failed: {e}")
                raise RuntimeError(f"Warmup failed: {e}") from e

    def generate_text_to_video(self, prompt: str, negative_prompt: str, width: int, height: int, 
                               fps: float, frames: int, seed: int, guidance_scale: float, 
                               inference_steps: int, output_path: str) -> Dict[str, Any]:
        """
        Generates video from a text prompt.
        """
        logger.info(f"Received text-to-video request: '{prompt}'")
        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError("TI2Vid pipeline is not loaded.")
            
            try:
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(frames, tiling_config)
                
                video, audio = self.ti2vid_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frames,
                    frame_rate=fps,
                    num_inference_steps=inference_steps,
                    video_guider_params=MultiModalGuiderParams(cfg_scale=guidance_scale),
                    audio_guider_params=MultiModalGuiderParams(cfg_scale=guidance_scale),
                    images=[],
                    tiling_config=tiling_config,
                )
                
                logger.info("Text-to-video generation successful.")
                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
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
            except Exception as e:
                logger.error(f"Error during text-to-video generation: {e}")
                raise RuntimeError(f"Generation failed: {e}") from e

    def generate_image_to_video(self, image_path: str, prompt: str, negative_prompt: str, width: int, height: int, 
                               fps: float, frames: int, seed: int, guidance_scale: float, 
                               inference_steps: int, output_path: str) -> Dict[str, Any]:
        """
        Generates video from a starting image and text prompt.
        """
        logger.info(f"Received image-to-video request: '{prompt}' with image: {image_path}")
        with self.lock:
            if not self.is_loaded or self.ti2vid_pipeline is None:
                raise RuntimeError("TI2Vid pipeline is not loaded.")
            
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Input image not found: {image_path}")

                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(frames, tiling_config)
                
                # Using ImageConditioningInput for the first frame
                images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
                
                video, audio = self.ti2vid_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=frames,
                    frame_rate=fps,
                    num_inference_steps=inference_steps,
                    video_guider_params=MultiModalGuiderParams(cfg_scale=guidance_scale),
                    audio_guider_params=MultiModalGuiderParams(cfg_scale=guidance_scale),
                    images=images,
                    tiling_config=tiling_config,
                )
                
                logger.info("Image-to-video generation successful.")
                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
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
            except Exception as e:
                logger.error(f"Error during image-to-video generation: {e}")
                raise RuntimeError(f"Generation failed: {e}") from e

    def generate_video_to_video(self, video_path: str, prompt: str, width: int, height: int, 
                               fps: float, frames: int, seed: int, output_path: str) -> Dict[str, Any]:
        """
        Generates video by transforming an existing video based on a prompt using IC-LoRA.
        """
        logger.info(f"Received video-to-video request: '{prompt}' with reference video: {video_path}")
        with self.lock:
            if not self.is_loaded or self.ic_lora_pipeline is None:
                raise RuntimeError("IC Lora Pipeline is not loaded. Cannot perform video-to-video.")
            
            try:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Input video not found: {video_path}")

                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(frames, tiling_config)
                
                video_conditioning = [(video_path, 1.0)]
                
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
                
                logger.info("Video-to-video generation successful.")
                encode_video(
                    video=video,
                    fps=int(fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
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
            except Exception as e:
                logger.error(f"Error during video-to-video generation: {e}")
                raise RuntimeError(f"Generation failed: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """
        Returns the health status of the LTXEngine.
        """
        return {
            "status": "healthy" if self.is_loaded else "not_loaded",
            "ti2vid_ready": self.ti2vid_pipeline is not None,
            "ic_lora_ready": self.ic_lora_pipeline is not None
        }

    def shutdown(self):
        """
        Shuts down the pipelines and frees up VRAM.
        """
        with self.lock:
            logger.info("Shutting down LTX engine...")
            self.ti2vid_pipeline = None
            self.ic_lora_pipeline = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared.")
                
            logger.info("Shutdown complete.")
