# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl

from flask import Blueprint, request, jsonify
from PIL import Image
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from src.utils.endpoint_util import required_param, within_range_inclusive, normalize_path
from src.utils.file_util import get_image_paths, get_image_save_path, get_timestamp
from src.utils.sdxl_util import get_sdxl_pipe, normalize_loras
from src.utils.cache_util import cache_set, cache_get
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
    "negative_prompt": payload.get("negative_prompt", None),
    "num_images": int(payload.get("num_images", 1)),
    "num_steps": int(payload.get("num_steps", 30)),
    "input_image_strength": int(payload.get("input_image_strength", 51)),
    "scale": payload.get("scale", 1.5)
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

    cache_key = "SDXL" + params["checkpoint_file_path"] + ",".join(lora["path"] for lora in params["loras"])
    sdxl_pipe = cache_get(cache_key)
    if sdxl_pipe is None:
      sdxl_pipe = get_sdxl_pipe(params["checkpoint_file_path"], params["loras"], "upscale")
      cache_set(cache_key, sdxl_pipe)

    for target in generation_targets:
      for _ in range(params["num_images"]):
        if not target["image_path"].lower().endswith(".png"):
          continue

        lower = target["image_path"].lower()
        if "_upscale" in lower:
          continue

        prompt = params["prompt"]
        if not prompt:
          json_path = os.path.splitext(target["image_path"])[0] + ".json"
          if os.path.isfile(json_path):
            try:
              with open(json_path, "r", encoding="utf-8") as jf:
                j = json.load(jf)
                if isinstance(j, dict) and j.get("prompt"):
                  prompt = j.get("prompt")
            except Exception:
              prompt = None

        # if still no prompt, skip this image
        if not prompt:
          continue

        negative_prompt = params["negative_prompt"]
        if not negative_prompt:
          json_path = os.path.splitext(target["image_path"])[0] + ".json"
          if os.path.isfile(json_path):
            try:
              with open(json_path, "r", encoding="utf-8") as jf:
                j = json.load(jf)
                if isinstance(j, dict) and j.get("negative_prompt"):
                  negative_prompt = j.get("negative_prompt")
            except Exception:
              negative_prompt = None

        prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = get_weighted_text_embeddings_sdxl(
          sdxl_pipe,
          prompt = f"{prompt}, high quality 8k resolution",
          neg_prompt = f"{negative_prompt}, low quality blurry"
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
