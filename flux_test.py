from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" # Temp
image = pipe(prompt).images[0]