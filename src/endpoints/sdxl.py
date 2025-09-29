from flask import Blueprint, request, jsonify
from PIL import Image
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from src.utils.endpoint_util import required_param, divisible_by_x, within_range_inclusive, create_seed, normalize_path
from src.utils.file_util import get_image_paths, get_image_save_path, get_timestamp
from src.utils.logger import log
from src.utils.sdxl_util import get_sdxl_pipe, normalize_loras
import gc
import json
import os
import random
import torch

sdxl_bp = Blueprint("sdxl", __name__, url_prefix="/api")

@sdxl_bp.route("/sdxl", methods=["POST"])
def sdxl():
  payload = request.get_json() or {}

  params = {
    "checkpoint_file_path": normalize_path(payload.get("checkpoint_file_path", None)),
    "loras": normalize_loras(payload.get("loras", []), 70),
    "prompt": payload.get("prompt", None),
    "negative_prompt": payload.get("negative_prompt", None),
    "seed": payload.get("seed", None),
    "width": int(payload.get("width", 1024)),
    "height": int(payload.get("height", 1024)),
    "num_images": int(payload.get("num_images", 1)),
    "num_steps": int(payload.get("num_steps", 60)),
    "output_folder_path": normalize_path(payload.get("output_folder_path", "output")),
    "output_image_prefix": str(payload.get("output_image_prefix", "")),
    "output_image_suffix": str(payload.get("output_image_suffix", "")),
    "input_image_path": payload.get("input_image_path"),
    "input_image_strength": int(payload.get("input_image_strength", 70))
  }

  required_param("prompt", params["prompt"])
  required_param("negative_prompt", params["negative_prompt"])

  divisible_by_x("height", params["height"], 8)
  divisible_by_x("width", params["width"], 8)

  within_range_inclusive("input_image_strength", params["input_image_strength"], 11, 100)

  image_paths = get_image_paths(params["input_image_path"])

  prompts = params["prompt"]
  if isinstance(prompts, str):
    prompts = [prompts]
  elif isinstance(prompts, list):
    prompts = [str(p) for p in prompts]
  else:
    prompts = [str(prompts)]

  sdxl_pipe = get_sdxl_pipe(params["checkpoint_file_path"], params["loras"], bool(params["input_image_path"]))
  saved_files = []

  try:
    generation_targets = []
    if image_paths:
      for img_path in image_paths:
        generation_targets.append({"image_path": img_path})
    else:
      generation_targets.append({"image_path": None})

    # Outer loop: iterate over each prompt in prompts array
    for prompt_text in prompts:
      prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = get_weighted_text_embeddings_sdxl(
        sdxl_pipe,
        prompt = prompt_text,
        neg_prompt = params["negative_prompt"]
      )

      for target in generation_targets:
        for _ in range(params["num_images"]):
          image_params = {
            **params,
            "seed": create_seed(params["seed"]),
            "prompt": prompt_text
          }
          gen = torch.Generator(device="cuda").manual_seed(image_params["seed"])

          if target["image_path"] is None:
            output = sdxl_pipe(
              prompt_embeds=prompt_embeds,
              pooled_prompt_embeds=pooled_prompt_embeds,
              negative_prompt_embeds=prompt_neg_embeds,
              negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
              num_inference_steps=image_params["num_steps"],
              height=image_params["height"],
              width=image_params["width"],
              num_images_per_prompt=1,
              generator=gen
            )
          else:
            init_image = Image.open(target["image_path"]).convert("RGB")
            target_size = (image_params["width"], image_params["height"])
            init_image = init_image.resize(target_size, resample=Image.LANCZOS)
            image_params["reference_image_path"] = target["image_path"]

            output = sdxl_pipe(
              prompt_embeds=prompt_embeds,
              pooled_prompt_embeds=pooled_prompt_embeds,
              negative_prompt_embeds=prompt_neg_embeds,
              negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
              image=init_image,
              strength=image_params["input_image_strength"] / 100,
              num_inference_steps=image_params["num_steps"],
              height=image_params["height"],
              width=image_params["width"],
              num_images_per_prompt=1,
              generator=gen
            )

          image = output.images[0]
          path = get_image_save_path(
            image_params["output_folder_path"],
            image_params["output_image_prefix"],
            image_params["output_image_suffix"]
          )
          image.save(path, format="PNG")
          saved_files.append(path)

          image_params["saved_image"] = path
          image_params["timestamp"] = get_timestamp()

          # save JSON metadata next to image with same base name but .json
          json_path = os.path.splitext(path)[0] + ".json"
          with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(image_params, jf, ensure_ascii=False, indent=2)

    return jsonify({"saved_files": saved_files}), 200

  finally:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
