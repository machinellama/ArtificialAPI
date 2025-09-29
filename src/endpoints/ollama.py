from flask import Blueprint, request, jsonify
from src.utils.ollama_util import call_ollama
from typing import Optional

ollama_bp = Blueprint("ollama", __name__, url_prefix="/api")

@ollama_bp.route("/ollama/prompt_variation", methods=["POST"])
def ollama():
  payload = request.get_json(silent=True) or {}

  # Parameters
  base_prompt = payload.get("base_prompt") or None
  variation_prompt = payload.get("variation_prompt") or None
  num_variations = int(payload.get("num_variations") or 1)
  ollama_url = payload.get("ollama_url", "http://localhost:11434/api/generate") or None
  ollama_model = payload.get("ollama_model", "gemma3:27b") or None

  if not base_prompt:
    return jsonify({"error": "`base_prompt` is required."}), 400
  if not variation_prompt:
    return jsonify({"error": "`variation_prompt` is required."}), 400

  variations = []

  # generate variations using the util function
  variations = generate_prompt_variations(
    base_prompt=base_prompt,
    variation_prompt=variation_prompt,
    count=num_variations,
    ollama_url=ollama_url,
    ollama_model=ollama_model
  )

  return jsonify({"base_prompt": base_prompt, "variation_prompt": variation_prompt, "variations": variations}), 200

def generate_prompt_variations(
  base_prompt: str,
  variation_prompt: Optional[str],
  count: int,
  ollama_url: str = "http://localhost:11434/api/generate",
  ollama_model: str = "gemma3:27b"
):
  """
  Generate `count` variations of base_prompt using an Ollama model.
  Returns a list of {"original": base_prompt, "new": <variation>} items.
  This function calls Ollama `count` times. It uses a schema to request a single
  string field `variation` from the model for easier parsing.
  """
  if not base_prompt:
    return []

  # Compose the prompt to instruct model: produce a single short variation per call
  instruction = variation_prompt or "Provide a concise, usable variation of the base prompt."
  # We'll create a small wrapper prompt sending base and instruction
  prompt_template = (
    "Base prompt:\n"
    "{base}\n\n"
    "Instruction:\n"
    "{instruction}\n\n"
    "Return only a JSON object with a single string field named \"variation\".\n"
    "The variation should be a rephrasing/alternative of the Base prompt suitable for image generation.\n"
    "Do not include any extra commentary."
  )

  schema = {
    "type": "object",
    "properties": {
      "variation": {"type": "string"}
    },
    "required": ["variation"]
  }

  results = []
  for i in range(max(0, int(count or 0))):
    composed = prompt_template.format(base=base_prompt, instruction=instruction)
    resp = call_ollama(
      ollama_url=ollama_url,
      model=ollama_model,
      prompt=composed,
      format_schema=schema,
      thinking=False,
      stream=False,
      keep_alive=None
    )

    variation_text = ""
    if isinstance(resp, dict) and "variation" in resp and isinstance(resp["variation"], str):
      variation_text = resp["variation"].strip()
    else:
      # Fallback: if resp has a single top-level string field or raw response, try to extract
      if isinstance(resp, dict):
        # find first string value
        for v in resp.values():
          if isinstance(v, str):
            variation_text = v.strip()
            break
      elif isinstance(resp, str):
        variation_text = resp.strip()

    results.append(variation_text)

  ## unload model before returning
  call_ollama(
    ollama_url=ollama_url,
    model=ollama_model,
    prompt="",
    format_schema=schema,
    thinking=False,
    stream=False,
    keep_alive=0
  )

  return results
