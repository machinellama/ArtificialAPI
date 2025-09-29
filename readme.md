# ArtificialAPI

There are a billion GUIs for image and video AI generation, but not many APIs that are easy to run and use with your own local scripts.

Built to enable scripting of automated media AI generation at mass scale.

Note: All code written and tested with 64GB RAM and Nvidia 4090 24GB VRAM. Your mileage may vary.

- [SDXL: text-to-image and image-to-image](#sdxl-images)
- [SDXL: upscale images](#sdxl-image-upscale)
- [Wan: text-to-video and image-to-video](#wan-video)
- [Ollama: prompt variations](#ollama-prompt-variations)
- More coming soon! contributions welcome

---

>Need help making your own AI tools or websites?
>
>Hire [dotfinally](https://dotfinally.com/en) and we can build any AI tool, extension, plugin, API, or website that you need.

---

## Prerequisites

- Nvidia GPU with CUDA v12.8
- (optional) Docker
- (optional) Ollama

## Run locally in Docker

- Start: `docker compose up -d`
- Stop: `docker compose down`
- Stop and clear: `docker compose down --volumes --remove-orphans`
- Stop, clear, rebuild, and restart: `docker compose down --volumes  --remove-orphans && docker compose up -d --build --force-recreate`

## Run locally

- Use an isolated Python environment (conda, venv, etc.)
- `conda create --name=aapi python=3.12.11`
- `conda activate aapi`
- `sudo apt update`
- `sudo apt install nvidia-cuda-toolkit`
  - Verify with `nvcc --version`
- `pip install -r requirements.txt`
- `pip install -U xformers --index-url https://download.pytorch.org/whl/cu128`
- `pip install git+https://github.com/xhinker/sd_embed.git@main`
- `python -m src.server`
  - Runs at `http://localhost:5700` by default
  - change host/port in server.py if needed

## Endpoints

### SDXL Images
Generate images using an SDXL checkpoint. Supports text-to-image and image-to-image.
- POST `http://localhost:5700/api/sdxl`
- Download a checkpoint, recommended from: [CivitAI SDXL Checkpoints](https://civitai.com/search/models?baseModel=SDXL%201.0&modelType=Checkpoint&sortBy=models_v9)
- Optionally download loras: [CivitAI SDXL Loras](https://civitai.com/search/models?baseModel=SDXL%201.0&modelType=LORA&sortBy=models_v9)

#### Parameters

| Name | Required | Type | Default | Description |
|------|----------|------|---------|-------------|
| checkpoint_file_path | Yes | string | — | Path to the SDXL checkpoint to load |
| loras | No | list | — | list of objects, with path and strength. Strength is between 1 and 100 inclusive. |
| prompt | Yes | string | — | Text prompt to generate the image, can be a single string or a list of strings. If list, each prompt will trigger a request with other params. |
| negative_prompt | Yes | string | — | Negative prompt to discourage content |
| seed | No | integer| — | Starting randomness of image. Empty or -1 will use random seeds; else given seed will be used for all images. |
| width | No | integer | 1024 | Output image width in pixels. Must be divisible by 8. |
| height | No | integer | 1024 | Output image height in pixels. Must be divisible by 8. |
| num_images | No | integer | 1 | Number of images to generate for the prompt. Each image will be saved separately. |
| num_steps | No | integer | 60 | Number of inference steps |
| output_folder_path | No | string | "output" | Folder to place saved images and metadata |
| output_image_prefix | No | string | — | Optional filename prefix for saved images |
| output_image_suffix | No | string | — | Optional filename suffix for saved images |
| input_image_path | No | string | — | Path to image or folder of images for image-to-image generation. If folder, then each image in the folder will trigger a separate generation request. |
| input_image_strength | No | integer | 70 | Amount of change applied to input image, must be between 1 and 100 inclusive. Higher number means more change. |

```
curl -X POST http://localhost:5700/api/sdxl \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_file_path": "models/sdxl/checkpoint/wow.safetensors",
    "loras": [
      {
        "path": "models/sdxl/lora/one.safetensors",
        "strength": 80
      },
      {
        "path": "models/sdxl/lora/two.safetensors",
        "strength": 50
      } 
    ],
    "prompt": "A fantasy landscape, vibrant colors, cinematic lighting",
    "negative_prompt": "blurry, cartoon",
    "seed": -1,
    "width": 1024,
    "height": 1024,
    "num_images": 2,
    "num_steps": 60,
    "output_folder_path": "output/sdxl_images",
    "output_image_prefix": "fantasy",
    "output_image_suffix": "v1",
    "input_image_path": "input/reference/test.png",
    "input_image_strength": 70
  }'
---
{
  "saved_files": [
    "output/sdxl_images/fantasy-1758590980-v1.png",
    "output/sdxl_images/fantasy-1758590999-v1.png"
  ]
}
```

### SDXL Image Upscale
Upscale images using an SDXL checkpoint.
- POST `http://localhost:5700/api/sdxl/upscale`
- Download a checkpoint, recommended from: [CivitAI SDXL Checkpoints](https://civitai.com/search/models?baseModel=SDXL%201.0&modelType=Checkpoint&sortBy=models_v9)
- Optionally download loras: [CivitAI SDXL Loras](https://civitai.com/search/models?baseModel=SDXL%201.0&modelType=LORA&sortBy=models_v9)

#### Parameters

| Name | Required | Type | Default | Description |
|------|----------|------|---------|-------------|
| checkpoint_file_path | Yes | string | — | Path to the SDXL checkpoint to load |
| loras | No | list | — | list of objects, with path and strength. Strength is between 1 and 100 inclusive. |
| upscale_path | Yes | string | — | Path to image or folder of images for upscaling. If folder, then each image in the folder will trigger a separate upscale request. |
| prompt | No | string | — | Text prompt to generate the image. If not provided, will look for .json file with prompt. |
| negative_prompt | Yes | string | — | Negative prompt to discourage content |
| num_images | No | integer | 1 | Number of images to generate for the prompt. Each image will be saved separately. |
| num_steps | No | integer | 30 | Number of inference steps |
| input_image_strength | No | integer | 51 | Amount of change applied to input image, must be between 1 and 100 inclusive. Higher number means more change. |
| scale | No | number | 1.5 | Scale for size of new upscaled image |

```
curl -X POST http://localhost:5700/api/sdxl/upscale \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_file_path": "models/sdxl/checkpoint/wow.safetensors",
    "loras": [
      {
        "path": "models/sdxl/lora/one.safetensors",
        "strength": 80
      },
      {
        "path": "models/sdxl/lora/two.safetensors",
        "strength": 50
      } 
    ],
    "upscale_path": "input/reference/images",
    "prompt": "A fantasy landscape, vibrant colors, cinematic lighting",
    "negative_prompt": "blurry, cartoon",
    "num_images": 2,
    "num_steps": 30,
    "input_image_strength": 51,
    "scale": 1.5
  }'
---
{
  "saved_files": [
    "output/sdxl_images/fantasy-1758590980-v1_upscaled_1758590983.png",
    "output/sdxl_images/fantasy-1758590982-v1_upscaled_1758590984.png"
  ]
}
```

### Wan Video
Generate videos using a Wan GGUF checkpoint. Supports text-to-video and image-to-video.
- POST `http://localhost:5700/api/wan`
- Download a Wan GGUF t2v or i2v model, recommended from: [Wan AIO GGUF](https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/tree/main/v10)
- Optionally download loras: [CivitAI Wan Loras](https://civitai.com/search/models?modelType=LORA&sortBy=models_v9&query=wan)

#### Parameters

| Name | Required | Type | Default | Description |
|------|----------|------|---------|-------------|
| gguf_path | Yes | string | — | URL or path to Wan GGUF model |
| loras | No | list | — | list of objects, with path and strength. Strength is between 1 and 100 inclusive. |
| prompt | No | string | — | Text prompt to generate the video. If not found, will look for .json file with prompt. |
| negative_prompt | Yes | string | — | Negative prompt to discourage content |
| seed | No | integer| — | Starting randomness of video. Empty or -1 will use random seeds; else given seed will be used for all videos. |
| width | No | integer | 480 | Output video width in pixels. Width - 1 must be divisible by 4. |
| height | No | integer | 720 | Output video height in pixels. Height - 1 must be divisible by 4. |
| num_videos | No | integer | 1 | Number of videos to generate for the prompt. Each video will be saved separately. |
| num_steps | No | integer | 4 | Number of inference steps |
| num_frames | No | integer | 81 | Number of total frames in the video (frames / fps = length of video) |
| fps | No | integer | 16 | Frames per second for generated video (frames / fps = length of video) |
| guidance_scale | No | integer | 1 | How closely to follow the prompt |
| output_folder_path | No | string | "output" | Folder to place saved videos and metadata |
| output_video_prefix | No | string | — | Optional filename prefix for saved videos |
| output_video_suffix | No | string | — | Optional filename suffix for saved videos |
| input_image_path | No | string | — | Path to image or folder of images for image-to-video generation. If folder, then each image in the folder will trigger a separate generation request. |

```
curl -X POST http://localhost:5700/api/wan \
  -H "Content-Type: application/json" \
  -d '{
    "gguf_path": "https://huggingface.co/befox/WAN2.2-14B-Rapid-AllInOne-GGUF/blob/main/v10/wan2.2-i2v-rapid-aio-v10-Q8_0.gguf",
    "loras": [
      {
        "path": "models/wan/lora/one.safetensors",
        "strength": 80
      },
      {
        "path": "models/wan/lora/two.safetensors",
        "strength": 50
      } 
    ],
    "prompt": "a cat flying through the sky",
    "negative_prompt": "blurry, cartoon, anime",
    "seed": -1,
    "width": 480,
    "height": 720,
    "num_videos": 2,
    "num_steps": 4,
    "num_frames": 81,
    "fps": 16,
    "guidance_scale": 1,
    "output_folder_path": "output/wan_videos",
    "output_video_prefix": "cat",
    "output_video_suffix": "v1",
    "input_image_path": "input/reference/test.png"
  }'
---
{
  "saved_files": [
    "output/wan_videos/cat-1758590980-v1.mp4",
    "output/wan_videos/cat-1758590999-v1.mp4"
  ]
}
```

### Ollama Prompt Variations
Returns variations of a given prompt, using ollama structured outputs.
- POST `http://localhost:5700/api/ollama/prompt_variation`
- Ollama must be running for this endpoint. We recommend running it locally in Docker: https://hub.docker.com/r/ollama/ollama

#### Parameters

| Name | Required | Type | Default | Description |
|------|----------|------|---------|-------------|
| base_prompt | Yes | string | — | Base prompt you want to vary |
| variation_prompt | Yes | string | — | Prompt to guide variations of the base prompt |
| num_variations | No | int | 1 | Number of different variations you want |
| ollama_url | No | string | "http://localhost:11434/api/generate" | URL path to the generate endpoint on your ollama instance |
| ollama_model | No | string | "gemma3:27b" | ollama model you want to use |

```
curl -X POST http://localhost:5700/api/ollama/prompt_variation \
  -H "Content-Type: application/json" \
  -d '{
    "base_prompt": "a cat wearing an astronaut suit and floating in a spaceship",
    "variation_prompt": "change what the cat is wearing, what it's doing, and it's location. Choose very random and unique variations. Must be a cat.",
    "num_variations": 3,
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "gemma3:27b"
  }'
---
{
  "base_prompt": "A cat wearing an astronaut suit and floating in a spaceship",
  "variation_prompt": "Change what the cat is wearing, what it's doing, and it's location. Choose very random and unique variations. Must be a cat.",
  "variations": [
    "A regal Tabby cat dressed as a Victorian-era royal, lifting a miniature planet above its head inside a giant teapot.",
    "A fluffy Persian cat dressed as a medieval knight, jousting with a rubber chicken in a giant bowl of petunias",
    "A fluffy calico cat in a superhero costume, flying through the clouds at night"
  ]
}

```

## License
This repo is MIT Licensed, but please check the licenses of any models you use.

Contributions welcome.
