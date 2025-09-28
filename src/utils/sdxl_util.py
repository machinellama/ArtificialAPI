# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, KDPM2DiscreteScheduler
from typing import Optional, List
import os
import torch

def get_sdxl_pipe(checkpoint_path, loras, is_img2img):
  if is_img2img:
    sdxl_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(checkpoint_path)
  else:
    sdxl_pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path)

  sdxl_pipe.scheduler = KDPM2DiscreteScheduler.from_config(sdxl_pipe.scheduler.config)

  adapter_names = []
  adapter_weights = []
  for lora in loras:
    sdxl_pipe.load_lora_weights(lora["path"], adapter_name=keep_alnum(lora["path"]))
    adapter_names.append(keep_alnum(lora["path"]))
    adapter_weights.append(lora["strength"] / 100)
  
  if loras:
    sdxl_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    sdxl_pipe.set_lora_device(adapter_names=adapter_names, device="cuda:0")

  sdxl_pipe.to(device="cuda", dtype=torch.float16)

  return sdxl_pipe

def normalize_loras(raw_loras, default_strength):
  if not raw_loras:
    return []
  normalized = []
  for idx, item in enumerate(raw_loras):
    if isinstance(item, str):
      path = item
      strength = default_strength
    elif isinstance(item, dict):
      path = item.get("path")
      if path is None:
        raise ValueError(f"loras[{idx}]: 'path' is required")
      strength = item.get("strength", default_strength)
    else:
      raise ValueError(f"loras[{idx}]: must be a string or object")

    if not isinstance(path, str) or not path:
      raise ValueError(f"loras[{idx}]: 'path' must be a non-empty string")
    try:
      strength = int(strength)
    except Exception:
      raise ValueError(f"loras[{idx}]: 'strength' must be an integer")
    if not (0 <= strength <= 100):
      raise ValueError(f"loras[{idx}]: 'strength' must be between 0 and 100")

    normalized.append({"path": path, "strength": strength})
  return normalized

def keep_alnum(s: str) -> str:
  """Return string with only ASCII letters and digits preserved."""
  return "".join(ch for ch in s if ch.isalnum())