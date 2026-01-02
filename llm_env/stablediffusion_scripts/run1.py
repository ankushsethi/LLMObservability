from diffusers import DiffusionPipeline
import torch

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\stable-diffusion"
out_path = r"llm_env\stablediffusion_scripts\out1.png"


pipe = DiffusionPipeline.from_pretrained(model_path, dtype=torch.float32, use_safetensors=True, variant="fp16")
pipe.to("cpu")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a red horse on moon surface"

images = pipe(prompt=prompt).images[0]
images.save(out_path)
print("Image saved at:", out_path)