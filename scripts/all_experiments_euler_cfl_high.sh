# BO
#done

# Base
# done

#OPRO
#./scripts/eval_template_new.sh G_openai euler_1d zero-shot cfl high
#./scripts/eval_template_new.sh G_Llama euler_1d zero-shot cfl high
#./scripts/eval_template_new.sh G_Qwen euler_1d zero-shot cfl high

#method-zeroshot
./scripts/eval_template_new.sh O_openai euler_1d zero-shot cfl high
./scripts/eval_template_new.sh O_Llama euler_1d zero-shot cfl high
./scripts/eval_template_new.sh O_Qwen euler_1d zero-shot cfl high

#method-iterative
./scripts/eval_template_new.sh O_openai euler_1d iterative cfl high
./scripts/eval_template_new.sh O_Llama euler_1d iterative cfl high
./scripts/eval_template_new.sh O_Qwen euler_1d iterative cfl high