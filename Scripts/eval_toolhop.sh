export PYTHONPATH=Code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

evaluation_file="evaluation_toolhop.py"
scenario="Direct"
file="ToolHop.jsonl"
model=" "
save_file="ToolHop-${model}-${scenario}.jsonl"

python3 Code/evaluation/${evaluation_file} --scenario ${scenario} --series qwen --model_path ${model} --input_file Data/jsonl/raw/${file} --output_file ${save_file} --enable_thinking --start_id 0 --end_id -1
