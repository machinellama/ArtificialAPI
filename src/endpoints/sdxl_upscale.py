# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl

from flask import Blueprint, request, jsonify
from PIL import Image
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from src.utils.endpoint_util import required_param, within_range_inclusive, normalize_path
from src.utils.file_util import get_image_paths, get_image_save_path, get_timestamp, get_json_value
from src.utils.sdxl_util import get_sdxl_pipe, normalize_loras
from src.utils.cache_util import cache_set, cache_get
from src.utils.logger import log
import json
import os
import torch
import time
import gc

sdxl_upscale_bp = Blueprint("sdxl_upscale", __name__, url_prefix="/api")

@sdxl_upscale_bp.route("/sdxl/upscale", methods=["POST"])
def sdxl_upscale():
  payload = request.get_json() or {}

  params = {
    "checkpoint_file_path": normalize_path(payload.get("checkpoint_file_path", None)),
    "loras": normalize_loras(payload.get("loras", []), 70),
    "upscale_path": payload.get("upscale_path"),
    "prompt": payload.get("prompt", None),
    "prompt_prefix": payload.get("prompt_prefix", None),
    "prompt_suffix": payload.get('prompt_suffix', None),
    "negative_prompt": payload.get("negative_prompt", None),
    "negative_prompt_prefix": payload.get("negative_prompt_prefix", None),
    "negative_prompt_suffix": payload.get("negative_prompt_suffix", None),
    "num_images": int(payload.get("num_images", 1)),
    "num_steps": int(payload.get("num_steps", 30)),
    "input_image_strength": int(payload.get("input_image_strength", 51)),
    "scale": payload.get("scale", 1.5),
    "force_upscale": payload.get("force_upscale", False)
  }

  required_param("checkpoint_file_path", params["checkpoint_file_path"])
  required_param("upscale_path", params["upscale_path"])

  within_range_inclusive("input_image_strength", params["input_image_strength"], 1, 100)

  image_paths = get_image_paths(params["upscale_path"])
  saved_files = []

  if not image_paths:
    return jsonify({"saved_files": saved_files}), 200

  try:
    generation_targets = []
    for img_path in image_paths:
      generation_targets.append({"image_path": img_path})

    log(f"Number of SDXL UPSCALE prompts to execute: {len(generation_targets)}")

    cache_key = "SDXL UPSCALE" + params["checkpoint_file_path"] + ",".join(f"{lora["path"]}-{lora["strength"]}" for lora in params["loras"])
    sdxl_pipe = cache_get(cache_key)
    if sdxl_pipe is None:
      sdxl_pipe, refiner_pipe = get_sdxl_pipe(params["checkpoint_file_path"], None, params["loras"], "upscale")
      cache_set(cache_key, sdxl_pipe)

    for ti, target in enumerate(generation_targets):
      log(f"SDXL UPSCALE prompt {ti + 1} / {len(generation_targets)}: {target["image_path"]}")

      for _ in range(params["num_images"]):
        if not target["image_path"].lower().endswith(".png"):
          continue

        lower = target["image_path"].lower()
        if "_upscale" in lower and not params["force_upscale"]:
          log("Skipping already upscaled image")
          continue

        prompt = params["prompt"]
        if not prompt:
          prompt = get_json_value(os.path.splitext(target["image_path"])[0], "prompt")

        # if still no prompt, skip this image
        if not prompt:
          log("Skipping, no prompt found")
          continue

        if params["prompt_prefix"]:
          prompt = params["prompt_prefix"] + prompt

        if params["prompt_suffix"]:
          prompt = prompt + params["prompt_suffix"]

        negative_prompt = params["negative_prompt"]
        if not negative_prompt:
          negative_prompt = get_json_value(os.path.splitext(target["image_path"])[0], "negative_prompt")

        if params["negative_prompt_prefix"]:
          negative_prompt = params["negative_prompt_prefix"] + negative_prompt

        if params["negative_prompt_suffix"]:
          negative_prompt = negative_prompt + params["negative_prompt_suffix"]

        prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = get_weighted_text_embeddings_sdxl(
          sdxl_pipe,
          prompt = prompt,
          neg_prompt = negative_prompt
        )

        base, _ = os.path.splitext(target["image_path"])
        ts = str(int(time.time()))
        out_path = f"{base}_upscaled_{ts}.png"

        img = Image.open(target["image_path"]).convert("RGB")
        orig_w, orig_h = img.size
        new_w, new_h = int(orig_w * params["scale"]), int(orig_h * params["scale"])
        big_img = img.resize((new_w, new_h), resample=Image.LANCZOS)

        output = sdxl_pipe(
          prompt_embeds=prompt_embeds,
          pooled_prompt_embeds=pooled_prompt_embeds,
          negative_prompt_embeds=prompt_neg_embeds,
          negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
          image=big_img,
          num_inference_steps=(params["num_steps"]),
          original_size=(orig_h, orig_w),
          target_size=(new_h, new_w),
          num_images_per_prompt=1,
          strength=params["input_image_strength"] / 100,
        )
        image = output.images[0]
        image.save(out_path)
        saved_files.append(out_path)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return jsonify({"saved_files": saved_files}), 200
  finally:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
