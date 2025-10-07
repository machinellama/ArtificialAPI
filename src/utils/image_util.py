from PIL import Image

def compute_dimensions_from_image(img_path, max_dim=720):
  try:
    with Image.open(img_path) as im:
      w, h = im.size
  except Exception:
    return None, None
  # if both <= max_dim, keep as-is
  if w <= max_dim and h <= max_dim:
    return w - (w % 16), h - (h % 16)
  # scale proportionally so the larger side becomes max_dim
  if w >= h:
    new_w = max_dim
    new_h = int(h * (max_dim / w))
  else:
    new_h = max_dim
    new_w = int(w * (max_dim / h))
  # ensure divisible by 16
  new_w = max(16, new_w - (new_w % 16))
  new_h = max(16, new_h - (new_h % 16))

  return new_w, new_h
