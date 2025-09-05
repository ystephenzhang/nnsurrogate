# BO
#./scripts/eval_template_new.sh BO_zero euler_1d zero-shot n_space medium
#./scripts/eval_template_new.sh BO_default euler_1d zero-shot n_space medium

# Base
# done

#OPRO
#./scripts/eval_template_new.sh G_openai euler_1d zero-shot n_space medium
#./scripts/eval_template_new.sh G_Llama euler_1d zero-shot n_space medium
#./scripts/eval_template_new.sh G_Qwen euler_1d zero-shot n_space medium

#method-zeroshot
#done
./scripts/eval_template_new.sh O_Llama euler_1d zero-shot n_space medium
./scripts/eval_template_new.sh O_Qwen euler_1d zero-shot n_space medium
#./scripts/eval_template_new.sh O_openai euler_1d iterative n_space medium

#method-iterative
#./scripts/eval_template_new.sh O_Llama euler_1d iterative n_space medium
#./scripts/eval_template_new.sh O_Qwen euler_1d iterative n_space medium
#./scripts/eval_template_new.sh O_openai euler_1d iterative n_space medium