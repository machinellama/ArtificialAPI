from flask import jsonify
from werkzeug.exceptions import BadRequest
import os
import random

def required_param(key, value):
  if not value:
    raise BadRequest(f"{key} is required")

def divisible_by_x(key, value, divisible):
  def floor_divisible_by_x(n: int) -> int:
    return n - (n % divisible)

  invalid_value = (value % divisible) != 0
  if invalid_value:
    adj_value = floor_divisible_by_x(value)

    raise BadRequest(f"{key} must be divisible by {divisible}, closest valid values below = {adj_value}")

def divisible_by_x_minus_one(key, value, divisible):
  def floor_divisible_by_x(n: int) -> int:
    return n - (n % divisible)

  invalid_value = ((value - 1) % divisible) != 0
  if invalid_value:
    adj_value = floor_divisible_by_x(value - 1)

    raise BadRequest(f"{key} - 1 must be divisible by {divisible}, closest valid values below = {adj_value + 1}")


def within_range_inclusive(key, value, min, max):
  if (value < min or value > max):
    raise BadRequest(f"{key} must be between {min} and {max} inclusive")

def create_seed(provided_seed):
  if provided_seed in (None, "", -1):
    return random.randrange(2**31)
  try:
    return int(provided_seed)
  except Exception:
    return abs(hash(str(provided_seed))) % (2**31)

def normalize_path(path):
  if path is None:
    return None

  return os.path.normpath(path)