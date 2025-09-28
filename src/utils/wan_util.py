
# https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/tree/main/Mega-v3

# https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
# https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
# https://huggingface.co/docs/diffusers/main/en/api/models/wan_transformer_3d
# https://huggingface.co/docs/diffusers/main/api/pipelines/wan#diffusers.WanPipeline

import torch
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan, GGUFQuantizationConfig, WanTransformer3DModel
from transformers import UMT5EncoderModel, CLIPVisionModel

import os
os.environ["TOKENIZERS_PARALLELISM"]="false"

def get_wan_pipe(
  gguf_path,
  loras,
  image_paths
):
  if bool(image_paths):
    transformer = WanTransformer3DModel.from_single_file(
      gguf_path,
      quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
      dtype=torch.bfloat16,
      config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
      subfolder="transformer",
      in_channels=36,
      patch_size=(1,2,2)
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
      "chatpig/umt5xxl-encoder-gguf",
      gguf_file="umt5xxl-encoder-q8_0.gguf",
      torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKLWan.from_pretrained(
      "callgg/wan-decoder",
      subfolder="vae",
      torch_dtype=torch.bfloat16
    )

    image_encoder = CLIPVisionModel.from_pretrained(
      "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
      subfolder="image_encoder",
      torch_dtype=torch.bfloat16
    )

    # group-offloading
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    apply_group_offloading(text_encoder,
      onload_device=onload_device,
      offload_device=offload_device,
      offload_type="block_level",
      num_blocks_per_group=4
    )
    apply_group_offloading(image_encoder,
      onload_device=onload_device,
      offload_device=offload_device,
      offload_type="block_level",
      num_blocks_per_group=4
    )
    transformer.enable_group_offload(
      onload_device=onload_device,
      offload_device=offload_device,
      offload_type="leaf_level",
      use_stream=True
    )

    pipeline = WanImageToVideoPipeline.from_pretrained(
      "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
      vae=vae,
      transformer=transformer,
      text_encoder=text_encoder,
      image_encoder=image_encoder,
      torch_dtype=torch.bfloat16
    )
  else:
    transformer = WanTransformer3DModel.from_single_file(
      gguf_url,
      quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
      dtype=torch.bfloat16,
      config="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
      subfolder="transformer",
      in_channels=16,
      patch_size=(1,2,2)
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
      "chatpig/umt5xxl-encoder-gguf",
      gguf_file="umt5xxl-encoder-q8_0.gguf",
      torch_dtype=torch.bfloat16,
    )

    vae = AutoencoderKLWan.from_pretrained(
      "callgg/wan-decoder",
      subfolder="vae",
      torch_dtype=torch.bfloat16
    )

    # group-offloading
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    apply_group_offloading(text_encoder,
      onload_device=onload_device,
      offload_device=offload_device,
      offload_type="block_level",
      num_blocks_per_group=4
    )
    transformer.enable_group_offload(
      onload_device=onload_device,
      offload_device=offload_device,
      offload_type="leaf_level",
      use_stream=True
    )

    pipeline = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      vae=vae,
      transformer=transformer,
      text_encoder=text_encoder,
      torch_dtype=torch.bfloat16
    )

  adapter_names = []
  adapter_weights = []
  for lora in loras:
    pipeline.load_lora_weights(lora["path"], adapter_name=keep_alnum(lora["path"]))
    adapter_names.append(keep_alnum(lora["path"]))
    adapter_weights.append(lora["strength"] / 100)

  if loras:
    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    pipeline.set_lora_device(adapter_names=adapter_names, device="cuda:0")

  pipeline.to(device="cuda", dtype=torch.bfloat16)

  return pipeline

def keep_alnum(s: str) -> str:
  """Return string with only ASCII letters and digits preserved."""
  return "".join(ch for ch in s if ch.isalnum())
