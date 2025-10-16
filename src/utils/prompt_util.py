import re
import itertools

def generate_prompt_variations(prompt: str, prompt_variations: dict) -> list:
  """
  Given a single prompt string and a dict of prompt_variations (keys -> list of values),
  returns a list of all prompt strings produced by replacing {{key}} with each value.
  If a key in prompt_variations is not present in the prompt, it's ignored.
  If prompt contains no keys, returns [prompt].
  """
  if not prompt or not prompt_variations:
    return [prompt]

  # find all unique keys used in the prompt like {{key}}
  keys_in_prompt = re.findall(r"\{\{(\w+)\}\}", prompt)
  if not keys_in_prompt:
    return [prompt]

  # keep only keys that have variations provided
  keys = [k for k in keys_in_prompt if k in prompt_variations and prompt_variations[k]]
  if not keys:
    return [prompt]

  # For stable ordering, preserve first-occurrence order of keys
  seen = []
  for k in keys_in_prompt:
    if k in keys and k not in seen:
      seen.append(k)
  keys = seen

  # build cartesian product of values for these keys
  value_lists = [prompt_variations[k] for k in keys]
  variations = []
  for combo in itertools.product(*value_lists):
    result = prompt
    for k, v in zip(keys, combo):
      result = re.sub(r"\{\{" + re.escape(k) + r"\}\}", str(v), result)
    variations.append(result)
  return variations

def prompt_contains_any(prompt: str, keywords: list[str]) -> bool:
  prompt_norm = prompt.lower().strip()
  for kw in keywords:
    if kw.lower().strip() in prompt_norm:
      return True
  return False
