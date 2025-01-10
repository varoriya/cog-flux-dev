# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline
)
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)
from huggingface_hub import hf_hub_download

MAX_IMAGE_SIZE = 1440
MODEL_CACHE = "FLUX.1-dev"
SAFETY_CACHE = "safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Flux txt2img Pipeline with new model")
        # ดึงโมเดลใหม่จาก Hugging Face
        model_path = hf_hub_download(
            repo_id="Varoriya/yuzakiflux",
            filename="RyuzakiMix_flux_Beta_fp16.safetensors"
        )
        self.txt2img_pipe = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE
        ).to("cuda")

        print("Loading Flux img2img pipeline")
        self.img2img_pipe = FluxImg2ImgPipeline(
            transformer=self.txt2img_pipe.transformer,
            scheduler=self.txt2img_pipe.scheduler,
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
        ).to("cuda")
        print("setup took: ", time.time() - start)

    @torch.amp.autocast('cuda')
    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to("cuda")
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

    def get_image(self, image: str):
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]

    @staticmethod
    def make_multiple_of_16(n):
        return ((n + 15) // 16) * 16

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=list(ASPECT_RATIOS.keys()),
            default="1:1"
        ),
        image: Path = Input(
            description="Input image for image-to-image mode. The aspect ratio of your output will match this image",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength (or denoising strength) when using image-to-image. 1.0 corresponds to full destruction of information in image.",
            ge=0, le=1, default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1, le=50, default=28,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0, le=10, default=3.5,
        ),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
    ) -> List[Path]:
        # ส่วนที่เหลือเหมือนโค้ดต้นฉบับ
