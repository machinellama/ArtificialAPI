from flask import Flask
from src.endpoints.sdxl import sdxl_bp
from src.endpoints.sdxl_upscale import sdxl_upscale_bp
from src.endpoints.ollama import ollama_bp
from src.endpoints.wan import wan_bp

def create_app(config_object = None):
  app = Flask(__name__, instance_relative_config = False)

  if config_object:
    app.config.from_object(config_object)

  app.register_blueprint(sdxl_bp)
  app.register_blueprint(sdxl_upscale_bp)
  app.register_blueprint(ollama_bp)
  app.register_blueprint(wan_bp)

  return app
