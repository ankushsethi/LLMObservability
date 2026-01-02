from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\stable-diffusion"
out_path = r"llm_env\stablediffusion_scripts\out2.png"

pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.feature_extractor = None

pipe.enable_attention_slicing()
pipe.vae.enable_tiling()
pipe.to("cpu")

prompt = "An astronaut riding a red horse on moon surface"

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=2.0
).images[0]

image.save(out_path)
print("Image saved at:", out_path)
