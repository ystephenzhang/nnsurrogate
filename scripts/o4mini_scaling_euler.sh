python inference/langchain_LLM.py -p custom_model -m OPRO_OpenAI_G -d euler_1d -t cfl -l high -z && \
python evaluation/euler_1d/eval.py -m OPRO_OpenAI_G -t cfl -l high -z && \

#python inference/langchain_LLM.py -p custom_model -m OPRO_OpenAI_S -d heat_1d -t n_space -l high -z
#python evaluation/heat_1d/eval.py -m OPRO_OpenAI_S -t n_space -l high -z

python inference/langchain_LLM.py -p custom_model -m iterative_baseline -d euler_1d -t cfl -l high -z && \
python evaluation/euler_1d/eval.py -m iterative_baseline -t cfl -l high -z