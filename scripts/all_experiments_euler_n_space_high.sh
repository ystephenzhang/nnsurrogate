# BO
./scripts/eval_template_new.sh BO_zero euler_1d zero-shot n_space high
./scripts/eval_template_new.sh BO_default euler_1d zero-shot n_space high

# Base
# done

#OPRO
./scripts/eval_template_new.sh G_openai euler_1d zero-shot n_space high
./scripts/eval_template_new.sh G_Llama euler_1d zero-shot n_space high
./scripts/eval_template_new.sh G_Qwen euler_1d zero-shot n_space high

#method-zeroshot
#done

#method-iterative
./scripts/eval_template_new.sh O_Llama euler_1d iterative n_space high
./scripts/eval_template_new.sh O_Qwen euler_1d iterative n_space high
./scripts/eval_template_new.sh O_openai euler_1d iterative n_space high