from diffusers.utils import export_to_video, load_image
from flask import Blueprint, request, jsonify
from src.utils.endpoint_util import required_param, divisible_by_x_minus_one, divisible_by_x, create_seed, normalize_path
from src.utils.file_util import get_image_paths, get_video_save_path, get_timestamp, get_json_value, concatenate_mp4s
from src.utils.wan_util import get_wan_pipe, load_loras
from src.utils.sdxl_util import normalize_loras
from src.utils.image_util import compute_dimensions_from_image
from src.utils.cache_util import cache_set, cache_get
import gc
import json
import os
import re
import torch
from PIL import Image
import cv2

wan_bp = Blueprint("wan", __name__, url_prefix="/api")

@wan_bp.route("/wan", methods=["POST"])
def wan():
  payload = request.get_json() or {}
  saved_files, pipeline = execute_wan(payload)

  return jsonify({ "saved_files": saved_files }), 200

@wan_bp.route("/wan/segments", methods=["POST"])
def wan_segments():
  payload = request.get_json() or {}
  segments = payload.get("segments", [])

  base_params = None
  all_files = []
  pipeline = None

  for i, segment in enumerate(segments):
    if i == 0:
      base_params = segment.copy()
    else:
      merged = base_params.copy()
      merged.update(segment)
      if not segment.get("input_image_path"):
        merged["input_image_path"] = None
      segment = merged

    if all_files and segment.get("input_image_path") is None:
      prev = all_files[-1]
      if prev.lower().endswith(".mp4") and os.path.exists(prev):
        # create output image path next to video
        base, _ = os.path.splitext(prev)
        img_path = f"{base}_lastframe.png"

        cap = cv2.VideoCapture(prev)
        # seek to last frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
          # fallback: read until end
          frame = None
          while True:
            ret, frame = cap.read()
            if not ret:
              break
          if frame is None:
            cap.release()
            raise RuntimeError(f"No frames found in {prev}")
        else:
          # set to last frame index (frame_count - 1)
          cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
          ret, frame = cap.read()
          if not ret or frame is None:
            # fallback: try one before last
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 2, 0))
            ret, frame = cap.read()
            if not ret:
              cap.release()
              raise RuntimeError(f"Failed to read last frame from {prev}")
        # write image as PNG
        cv2.imwrite(img_path, frame)
        cap.release()

        segment["input_image_path"] = img_path

    segment["pipeline"] = pipeline
    segment["segment_index"] = i

    saved_files, pipeline = execute_wan(segment)

    if saved_files:
      all_files.extend(saved_files)

  mp4s = [f for f in all_files if f.lower().endswith(".mp4")]
  if len(mp4s) > 1:
    combined_file_path = get_video_save_path(
      segments[0].get("output_folder_path"),
      segments[0].get("output_video_prefix"),
      segments[0].get("output_video_suffix")
    )
    concat_output_path = concatenate_mp4s(mp4s, combined_file_path)
    all_files.extend(concat_output_path)

  return jsonify({"all_files": all_files}), 200

def execute_wan(payload):
  params = {
    "gguf_path": payload.get("gguf_path", None),
    "loras": normalize_loras(payload.get("loras", []), 70),
    "prompt": payload.get("prompt", None),
    "negative_prompt": payload.get("negative_prompt", None),
    "seed": payload.get("seed", None),
    "width": payload.get("width", None),
    "height": payload.get("height", None),
    "num_videos": int(payload.get("num_videos", 1)),
    "num_steps": int(payload.get("num_steps", 4)),
    "num_frames": int(payload.get("num_frames", 81)),
    "fps": int(payload.get("fps", 16)),
    "guidance_scale": int(payload.get("guidance_scale", 1)),
    "output_folder_path": normalize_path(payload.get("output_folder_path", "output")),
    "output_video_prefix": str(payload.get("output_video_prefix", "")),
    "output_video_suffix": str(payload.get("output_video_suffix", "")),
    "input_image_path": payload.get("input_image_path"),
    "segment_index": payload.get("segment_index", -1)
  }

  required_param("gguf_path", params["gguf_path"])

  divisible_by_x_minus_one("num_frames", params["num_frames"], 4)

  # width/height may be None; validate only when provided
  if params["height"] is not None:
    divisible_by_x("height", int(params["height"]), 16)
    params["height"] = int(params["height"])
  if params["width"] is not None:
    divisible_by_x("width", int(params["width"]), 16)
    params["width"] = int(params["width"])

  image_paths = get_image_paths(params["input_image_path"])

  saved_files = []

  try:
    pipeline = None
    if payload.get("pipeline"):
      pipeline = payload.get("pipeline")
      load_loras(pipeline, params["loras"], params["segment_index"])
    else:
      cache_key = "WAN" + ",".join(lora["path"] for lora in params["loras"]) + str(bool(image_paths))
      pipeline = cache_get(cache_key)
      if pipeline is None:
        pipeline = get_wan_pipe(
          params["gguf_path"],
          params["loras"],
          bool(image_paths),
          params["segment_index"]
        )
        cache_set(cache_key, pipeline)

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
          base = os.path.splitext(target["image_path"])[0]
          prompt = get_json_value(base, "prompt")

        if not prompt and not params["prompt"]:
          continue # skip if no prompt

        negative_prompt = params["negative_prompt"]
        if not negative_prompt and target["image_path"]:
          base = os.path.splitext(target["image_path"])[0]
          negative_prompt = get_json_value(base, "negative_prompt")

        # compute per-target width/height if not provided
        width = params["width"]
        height = params["height"]
        if target["image_path"] and (width is None or height is None):
          img_w, img_h = compute_dimensions_from_image(target["image_path"], max_dim=720)
          if img_w and img_h:
            # use image dims for whichever missing, otherwise keep provided
            if width is None:
              width = img_w
            if height is None:
              height = img_h
        # fallback defaults if still None
        if width is None:
          width = 480
        if height is None:
          height = 720
        # ensure divisible by 16
        width = int(width) - (int(width) % 16)
        height = int(height) - (int(height) % 16)

        video_params = {
          **params,
          "prompt": prompt or params["prompt"],
          "negative_prompt": negative_prompt or params["negative_prompt"],
          "seed": create_seed(params["seed"]),
          "width": width,
          "height": height
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

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return saved_files, pipeline

  finally:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
