# https://docs.ollama.com/api#generate-a-completion

from flask import request, jsonify
import json
import requests
from typing import Optional

def call_ollama(
  ollama_url: str,
  model: str,
  prompt: str,
  format_schema = None,
  thinking: bool = True,
  stream: bool = False,
  timeout: int = 300,
  keep_alive: int = None
):
  """
  Call Ollama /api/generate endpoint with a prompt and optional format schema.
  Returns parsed JSON response (raises for HTTP errors or JSON parsing issues).
  """
  payload = {
    "model": model,
    "prompt": prompt,
    "thinking": thinking,
    "stream": stream
  }
  if format_schema is not None:
    payload["format"] = format_schema
  if keep_alive is not None:
    payload["keep_alive"] = keep_alive

  headers = {"Content-Type": "application/json"}
  resp = requests.post(ollama_url, headers=headers, json=payload, timeout=timeout)
  resp.raise_for_status()
  resp_json = resp.json()

  ## unload model before returning
  requests.post(
    ollama_url,
    headers=headers,
    json={
      "model": model,
      "keep_alive": 0
    },
    timeout=timeout
  )

  # Ollama returns a field like resp_json["response"] which is a stringified JSON when using format
  # If format was provided try to parse resp_json["response"], otherwise return entire resp_json
  if isinstance(resp_json, dict) and "response" in resp_json:
    try:
      return json.loads(resp_json["response"])
    except (TypeError, ValueError):
      # If it's not JSON, return the raw response field
      return {"response": resp_json["response"]}
  return resp_json
