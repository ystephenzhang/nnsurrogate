python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d heat_1d -t n_space -l high -z --resume
python evaluation/heat_1d/eval.py -m llama3.2_3b_base -t n_space -l high -z

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d heat_1d -t n_space -l high --resume
python evaluation/heat_1d/eval.py -m llama3.2_3b_base -t n_space -l high

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l medium --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l medium

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l medium -z --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l medium -z

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l high --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l high

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l high -z --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l high -z

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l low --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l low

python inference/langchain_LLM.py -p custom_model -m llama3.2_3b_base -d euler_1d -t cfl -l low -z --resume
python evaluation/euler_1d/eval.py -m llama3.2_3b_base -t cfl -l low -z