import os
import time
import tzlocal
import re
from datetime import datetime

def get_image_paths(input_image_path):
  image_paths = []
  if input_image_path is not None:
    def _abs_input_path(p):
      # normalize the provided path (leave absolute paths unchanged)
      return os.path.normpath(p)

    # Normalize payload value into a list
    candidates = []
    if isinstance(input_image_path, list):
      candidates = input_image_path
    else:
      candidates = [input_image_path]

    for cand in candidates:
      if not isinstance(cand, str):
        continue
      abs_path = _abs_input_path(cand)
      # If path is a directory, collect all .png files inside (non-recursive)
      if os.path.isdir(abs_path):
        try:
          for fn in sorted(os.listdir(abs_path)):
            if fn.lower().endswith(".png"):
              image_paths.append(os.path.join(abs_path, fn))
        except Exception:
          # skip folder if cannot read
          continue
      else:
        # If file exists and is a png, add it.
        if os.path.isfile(abs_path) and abs_path.lower().endswith(".png"):
          image_paths.append(abs_path)
        # else skip silently (non-existent or non-png)

  print(f"image_paths: {image_paths}")
  return image_paths

def get_image_save_path(folder, prefix, suffix):
  parts = []

  if prefix:
    parts.append(prefix)
  ts = str(int(time.time()))
  parts.append(ts)
  if suffix:
    parts.append(suffix)
  filename = "-".join(parts) + ".png"

  os.makedirs(folder, exist_ok=True)
  path = os.path.join(folder, filename)

  return path

def get_video_save_path(folder, prefix, suffix):
  parts = []

  if prefix:
    parts.append(prefix)
  ts = str(int(time.time()))
  parts.append(ts)
  if suffix:
    parts.append(suffix)
  filename = "-".join(parts) + ".mp4"

  os.makedirs(folder, exist_ok=True)
  path = os.path.join(folder, filename)

  return path

def get_timestamp():
  local_tz = tzlocal.get_localzone()
  now = datetime.now(local_tz)
  iso_ts = now.isoformat(timespec='milliseconds')
  tz_abbr = now.tzname()
  timestamp_str = f"{iso_ts} ({tz_abbr})" if tz_abbr else iso_ts

  return timestamp_str

def get_json_value(base, key):
  m = re.match(r"^(.*)_upscaled_\d+$", base)

  if m:
    json_path = m.group(1) + ".json"
  else:
    json_path = base + ".json"

  if os.path.isfile(json_path):
    try:
      with open(json_path, "r", encoding="utf-8") as jf:
        j = json.load(jf)
        if isinstance(j, dict) and j.get(key):
          value = j.get(key)
    except Exception:
      value = None

  return value
