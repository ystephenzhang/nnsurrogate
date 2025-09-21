
# Tolerance experiment configs - iterative only  
#./eval_template_new.sh heat_1d O_Llama_relaxed_t05 zero-shot custom_model low
#./eval_template_new.sh heat_1d O_Llama_relaxed_t05 iterative custom_model low

#./eval_template_new.sh heat_1d O_Qwen_relaxed_t05 zero-shot custom_model low
#./eval_template_new.sh heat_1d O_Qwen_relaxed_t05 iterative custom_model low

#./eval_template_new.sh heat_1d O_Llama_relaxed_t05 zero-shot custom_model all resume
./eval_template_new.sh heat_1d O_Llama_relaxed_t03 iterative custom_model all

#./eval_template_new.sh heat_1d O_Qwen_relaxed_t05 zero-shot custom_model all resume
./eval_template_new.sh heat_1d O_Qwen_relaxed_t03 iterative custom_model all 

