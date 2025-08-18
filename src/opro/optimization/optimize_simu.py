import datetime
import functools
import getpass
import json
import os
import re
import sys
import itertools

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import hydra
from omegaconf import DictConfig
import numpy as np
import openai

from opro import prompt_utils

sys.path.append('/home/ubuntu/dev/src')
from general_utils import Verifier, INFO, complete_params_with_initial
from simu_utils import * 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ============== set optimization experiment configurations ================
    num_points = cfg.optimization.num_points
    num_steps = cfg.optimization.num_steps
    max_num_pairs = cfg.optimization.max_num_pairs
    num_decimals = cfg.optimization.num_decimals
    num_starting_points = cfg.optimization.num_starting_points
    num_decode_per_step = cfg.optimization.num_decode_per_step
    problem = cfg.problem
    task = cfg.task
    profile = cfg.profile
    log_real_value = cfg.get("log_real_value", False)
    evaluation_backend = cfg.get("evaluation_backend", "surrogate")

    # ================ load LLM settings ===================
    optimizer_llm_name = cfg.llm.optimizer
    openai_api_key = cfg.llm.openai_api_key

    if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4", "deepseek-chat", "deepseek-reasoner", "o4-mini"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        import os
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif optimizer_llm_name in {"local"}:
        raise NotImplementedError

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        OPRO_ROOT_PATH,
        "outputs",
        "optimization-results",
        f"simu-{problem}-{task}-{profile}-o-{optimizer_llm_name}-{evaluation_backend}-{datetime_str}/",
    )
    os.makedirs(save_folder)
    print(f"result directory:\n{save_folder}")

    # ====================== utility functions ============================
    if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4", "deepseek-chat", "deepseek-reasoner", "o4-mini"}:
        optimizer_gpt_max_decode_steps = 1024
        optimizer_gpt_temperature = 1.0

        optimizer_llm_dict = dict()
        optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
        optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
        optimizer_llm_dict["batch_size"] = 1
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=optimizer_llm_name,
            max_decode_steps=optimizer_gpt_max_decode_steps,
            temperature=optimizer_gpt_temperature,
        )

    elif optimizer_llm_name in {"local"}: #local model
        raise NotImplementedError
    # ====================== try calling the servers ============================
    print("\n======== testing the optimizer server ===========")
    #optimizer_test_output = call_optimizer_server_func(
    #    "Does the sun rise from the north? Just answer yes or no.",
    #    temperature=1.0,
    #)
    #print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the optimizer server.")
    print("\n=================================================")
  
    # ================= generate the ground truth trace =====================
    verifier = Verifier(problem, task)
    init_sols = sample_sols(problem, task, verifier, profile, num_starting_points, sampler="random") # [(params: {}, score: x)]
    gt_sol = get_dummy(cfg.profile, verifier)
    # ====================== run optimization ============================
    configs_dict = {
        "num_starting_points": num_starting_points,
        "num_decode_per_step": num_decode_per_step,
        "optimizer_llm_configs": optimizer_llm_dict,
        "data": {
            "ground truth solution": gt_sol,
            "profile": cfg.profile,
            "problem": cfg.problem,
            "task":cfg.task
        },
        "init_sols": init_sols,
        "num_steps": num_steps,
        "max_num_pairs": max_num_pairs,
        "num_decimals": num_decimals,
    }
    
    old_value_pairs_set = set()
    old_value_pairs_with_i_step = []  # format: [(trace, dis = f(trace), i_step)]
    meta_prompts_dict = dict()  # format: {i_step: meta_prompt}
    raw_outputs_dict = dict()  # format: {i_step: raw_outputs}

    #for sol in init_sols:
    #    #pdb.set_trace()
    #    old_value_pairs_set.add((str(sol[0]), sol[1]))
    #    old_value_pairs_with_i_step.append((str(str(sol[0])), sol[1], -1))

    print("\n================ run optimization ==============")
    print(f"initial points: {[item[0] for item in old_value_pairs_set]}")
    print(f"initial values: {[item[-1] for item in old_value_pairs_set]}")
    results_json_path = os.path.join(save_folder, "results.json")
    print(f"saving results to\n{results_json_path}")

    for i_step in range(num_steps):
        print(f"\nStep {i_step}:")
        meta_prompt = gen_meta_prompt(
            old_value_pairs_set,
            profile,
            problem,
            task,
            max_num_pairs=max_num_pairs,
        )
        print("\n=================================================")
        print(f"meta_prompt:\n{meta_prompt}")
        meta_prompts_dict[i_step] = meta_prompt
        raw_outputs = []
        parsed_outputs = []
        
        while len(parsed_outputs) < num_decode_per_step:
            raw_output = call_optimizer_server_func(meta_prompt)
            for string in raw_output:
                print("\n=================================================")
                print("raw output:\n", string)
                #try:
                extracted = extract_string(string, task)
                if extracted is not None and "value" in extracted:
                    # Convert the extracted value to proper parameter format
                    params = {task: extracted["value"]}
                    # Complete params with initial values for other parameters
                    params = complete_params_with_initial(params, problem)
                    
                    # Validate parameter is in range
                    score = evaluate(params, verifier, profile, backend=evaluation_backend, surrogate_config=cfg.surrogate_config)
                    #score = evaluate(params, verifier, profile, backend="ground_truth")

                    if score > 0:  # Only accept positive scores
                        parsed_outputs.append(params)
                        raw_outputs.append(string)
                #except Exception as e:
                #    print(f"Error processing output: {e}")
                #    pass
        print("\n=================================================")
        print(f"proposed points: {parsed_outputs}")
        raw_outputs_dict[i_step] = raw_outputs

        # evaluate the values of proposed and rounded outputs
        single_step_values = []
        for params in parsed_outputs:
            score = evaluate(params, verifier, profile, backend=evaluation_backend)
            if log_real_value:
                if evaluation_backend == "surrogate":
                    real_value = evaluate(params, verifier, profile, backend="ground_truth")
                else:
                    real_value = score
            #score = evaluate(params, verifier, profile, backend="ground_truth")
            single_step_values.append(score)
            old_value_pairs_set.add((str(params), score))
            old_value_pairs_with_i_step.append((params, score, real_value, i_step))
        print(f"single_step_values: {single_step_values}")
        print("dummy solution", gt_sol)

        final_value_pair = max(old_value_pairs_with_i_step, key=lambda x: x[1])
        final_pair_score = evaluate(final_value_pair[0], verifier, profile, backend="ground_truth")
        # ====================== save results ============================
        results_dict = {
            "meta_prompts": meta_prompts_dict,
            "raw_outputs": raw_outputs_dict,
            "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
            "chosen_value_pair": final_value_pair,
            "final_score": final_pair_score
        }
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    main()