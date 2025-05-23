
# Experiment of "DiVaTe Benchmark"

This project supports generating imagesfrom prompts using multiple models: **FLUX**/**Midjourney**/**GPT-image-1/Deepfloyd/SANA/Stable Diffusion3**/**Stable Diffusion2**/**AnyText**/**TextDiffuser2**.
This project also supports postprocessing and evaluation. 




## How to Use

1. Download requirements from requirements.txt  (pip install -r requirements.txt)
2. Download benchmark prompts from (https://github.com/anonymous-author-submission/DiVaTe_Bench) (Here, however, we already have them in make_prompts directory, so no need to do this)
3. Follow test_script.sh (e.g. python3 main.py --model sana --prompt_path make_prompts/${category}.txt --model_path ${model_path}/Sana_Sprint_1.6B_1024px_diffusers --save_path ${results_dir}
)
