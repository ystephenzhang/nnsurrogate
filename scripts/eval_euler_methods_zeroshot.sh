#./scripts/eval_template_new.sh O_openai euler_1d zero-shot medium
#./scripts/eval_template_new.sh O_Qwen euler_1d zero-shot medium
#./scripts/eval_template.sh O_Llama euler_1d zero-shot medium

#./scripts/eval_template_new.sh O_openai euler_1d zero-shot cfl high
#./scripts/eval_template_new.sh O_Qwen euler_1d zero-shot cfl high
#./scripts/eval_template_new.sh O_Llama euler_1d zero-shot cfl high

./scripts/eval_template_new.sh O_openai heat_1d zero-shot n_space high
./scripts/eval_template_new.sh O_Llama heat_1d zero-shot n_space high
./scripts/eval_template_new.sh O_Qwen heat_1d zero-shot n_space high

#./scripts/eval_template.sh OPRO_Llama_S euler_1d_cfl iterative medium
#./scripts/eval_template.sh OPRO_Qwen_S euler_1d_cfl iterative medium
#./scripts/eval_template.sh OPRO_OpenAI_S euler_1d_cfl iterative medium
