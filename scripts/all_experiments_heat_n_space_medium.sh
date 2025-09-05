# BO
#./scripts/eval_template_new.sh BO_zero heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh BO_default heat_1d zero-shot n_space medium

# Base
#./scripts/eval_template_new.sh o4-mini heat_1d zero-shot n_space medium openai
./scripts/eval_template_new.sh llama3.2_3b_base heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh qwen3_8b heat_1d zero-shot n_space medium

#./scripts/eval_template_new.sh o4-mini heat_1d iterative n_space medium openai
./scripts/eval_template_new.sh llama3.2_3b_base heat_1d iterative n_space medium
./scripts/eval_template_new.sh qwen3_8b heat_1d iterative n_space medium

#OPRO
#./scripts/eval_template_new.sh G_openai heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh G_Llama heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh G_Qwen heat_1d zero-shot n_space medium

#method-zeroshot
#./scripts/eval_template_new.sh O_openai heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh O_Llama heat_1d zero-shot n_space medium
#./scripts/eval_template_new.sh O_Qwen heat_1d zero-shot n_space medium

#method-iterative
#./scripts/eval_template_new.sh O_openai heat_1d iterative n_space medium
#./scripts/eval_template_new.sh O_Llama heat_1d iterative n_space medium
#./scripts/eval_template_new.sh O_Qwen heat_1d iterative n_space medium