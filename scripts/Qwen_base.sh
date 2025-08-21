python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d heat_1d -t n_space -l high -z --resume
python evaluation/heat_1d/eval.py -m qwen3_8b -t n_space -l high -z

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d heat_1d -t n_space -l high --resume
python evaluation/heat_1d/eval.py -m qwen3_8b -t n_space -l high

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l medium --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l medium

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l medium -z --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l medium -z

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l high --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l high

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l high -z --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l high -z

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l low --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l low

python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -l low -z --resume
python evaluation/euler_1d/eval.py -m qwen3_8b -t cfl -l low -z