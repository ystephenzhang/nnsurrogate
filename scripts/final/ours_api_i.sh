
# Tolerance experiment configs - iterative only
#./eval_template_new.sh heat_1d ab_openai_init_refined_t03 iterative custom_model
./eval_template_new.sh heat_1d ab_openai_init_refined_t05 iterative custom_model  
#./eval_template_new.sh heat_1d ab_openai_init_refined_t07 iterative custom_model 

#./eval_template_new.sh euler_1d ab_openai_init_refined_t03 iterative custom_model resume
./eval_template_new.sh euler_1d ab_openai_init_refined_t05 iterative custom_model resume
#./eval_template_new.sh euler_1d ab_openai_init_refined_t07 iterative custom_model resume

#./eval_template_new.sh ns_2d ab_openai_init_refined_t03 iterative custom_model
./eval_template_new.sh ns_transient_2d ab_openai_init_refined_t05 iterative custom_model  
#./eval_template_new.sh ns_2d ab_openai_init_refined_t07 iterative custom_model