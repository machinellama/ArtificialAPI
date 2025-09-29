from diffusers.utils import export_to_video, load_image
from flask import Blueprint, request, jsonify
from src.utils.endpoint_util import required_param, divisible_by_x_minus_one, create_seed, normalize_path
from src.utils.file_util import get_image_paths, get_video_save_path, get_timestamp
from src.utils.wan_util import get_wan_pipe
from src.utils.sdxl_util import normalize_loras
import gc
import json
import os
import torch

wan_bp = Blueprint("wan", __name__, url_prefix="/api")

@wan_bp.route("/wan", methods=["POST"])
def wan():
  payload = request.get_json() or {}

  params = {
    "gguf_path": payload.get("gguf_path", None),
    "loras": normalize_loras(payload.get("loras", []), 70),
    "prompt": payload.get("prompt", None),
    "negative_prompt": payload.get("negative_prompt", None),
    "seed": payload.get("seed", None),
    "width": int(payload.get("width", 480)),
    "height": int(payload.get("height", 720)),
    "num_videos": int(payload.get("num_videos", 1)),
    "num_steps": int(payload.get("num_steps", 4)),
    "num_frames": int(payload.get("num_frames", 81)),
    "fps": int(payload.get("fps", 16)),
    "guidance_scale": int(payload.get("guidance_scale", 1)),
    "output_folder_path": normalize_path(payload.get("output_folder_path", "output")),
    "output_video_prefix": str(payload.get("output_video_prefix", "")),
    "output_video_suffix": str(payload.get("output_video_suffix", "")),
    "input_image_path": str(payload.get("input_image_path", "")),
  }

  required_param("gguf_path", params["gguf_path"])
  required_param("negative_prompt", params["negative_prompt"])

  divisible_by_x_minus_one("num_frames", params["num_frames"], 4)
  
  image_paths = get_image_paths(params["input_image_path"])

  saved_files = []

  try:
    pipeline = get_wan_pipe(
      params["gguf_path"],
      params["loras"],
      image_paths
    )

    generation_targets = []
    if image_paths:
      for img_path in image_paths:
        generation_targets.append({"image_path": img_path})
    else:
      generation_targets.append({"image_path": None})

    for target in generation_targets:
      for _ in range(params["num_videos"]):
        # determine prompt: use request prompt if provided, otherwise look for same-name .json with prompt field
        prompt = params["prompt"]
        if not prompt and target["image_path"]:
          # if filename contains _upscaled_{ts}, strip that part to find original json next to original image
          base = os.path.splitext(target["image_path"])[0]
          # remove trailing _upscaled_<digits> if present
          import re
          m = re.match(r"^(.*)_upscaled_\d+$", base)
          if m:
            json_path = m.group(1) + ".json"
          else:
            json_path = base + ".json"

          if os.path.isfile(json_path):
            try:
              with open(json_path, "r", encoding="utf-8") as jf:
                j = json.load(jf)
                if isinstance(j, dict) and j.get("prompt"):
                  prompt = j.get("prompt")
            except Exception:
              prompt = None

        # if still no prompt and no image-based prompt, require top-level prompt
        if not prompt and not params["prompt"]:
          # skip this target if no prompt available
          continue

        video_params = {
          **params,
          "prompt": prompt or params["prompt"],
          "seed": create_seed(params["seed"]),
        }
        gen = torch.Generator(device="cuda").manual_seed(video_params["seed"])

        image_arg = None
        if target["image_path"]:
          image_arg = load_image(target["image_path"])
          video_params["image"] = target["image_path"]

        call_kwargs = dict(
          prompt=video_params["prompt"],
          negative_prompt=video_params["negative_prompt"],
          height=video_params["height"],
          width=video_params["width"],
          num_frames=video_params["num_frames"],
          guidance_scale=video_params["guidance_scale"],
          num_inference_steps=video_params["num_steps"],
          generator=gen
        )

        if image_arg is not None:
          call_kwargs["image"] = image_arg
        output = pipeline(**call_kwargs).frames[0]

        path = get_video_save_path(
          video_params["output_folder_path"],
          video_params["output_video_prefix"],
          video_params["output_video_suffix"]
        )
        export_to_video(output, path, fps=video_params["fps"])
        saved_files.append(path)

        video_params["saved_video"] = path
        video_params["timestamp"] = get_timestamp()

        # save JSON metadata next to video with same base name but .json
        json_path_out = os.path.splitext(path)[0] + ".json"
        with open(json_path_out, "w", encoding="utf-8") as jf:
          json.dump(video_params, jf, ensure_ascii=False, indent=2)

    return jsonify({ "saved_files": saved_files }), 200

  finally:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
