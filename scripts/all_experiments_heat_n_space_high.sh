# BO
# done

# Base
# done

#OPRO
#./scripts/eval_template_new.sh G_openai heat_1d zero-shot n_space high
#./scripts/eval_template_new.sh G_Llama heat_1d zero-shot n_space high
#./scripts/eval_template_new.sh G_Qwen heat_1d zero-shot n_space high

#method-zeroshot
#./scripts/eval_template_new.sh O_openai heat_1d zero-shot n_space high
#./scripts/eval_template_new.sh O_Llama heat_1d zero-shot n_space high
#./scripts/eval_template_new.sh O_Qwen heat_1d zero-shot n_space high

#method-iterative
./scripts/eval_template_new.sh O_openai heat_1d iterative n_space high
./scripts/eval_template_new.sh O_Llama heat_1d iterative n_space high
./scripts/eval_template_new.sh O_Qwen heat_1d iterative n_space high