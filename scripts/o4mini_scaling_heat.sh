python inference/langchain_LLM.py -p custom_model -m OPRO_OpenAI_G -d heat_1d -t n_space -l high -z && \
python evaluation/heat_1d/eval.py -m OPRO_OpenAI_G -t n_space -l high -z && \

python inference/langchain_LLM.py -p custom_model -m OPRO_OpenAI_S -d heat_1d -t n_space -l high -z
python evaluation/heat_1d/eval.py -m OPRO_OpenAI_S -t n_space -l high -z

python inference/langchain_LLM.py -p custom_model -m iterative_baseline -d heat_1d -t n_space -l high -z --resume
python evaluation/heat_1d/eval.py -m iterative_baseline -t n_space -l high -z