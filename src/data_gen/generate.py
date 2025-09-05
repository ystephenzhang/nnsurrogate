import sys
sys.path.append("/home/ubuntu/dev/src")
from general_utils import Verifier
import general_utils
import json
import numpy as np
import os
import copy
import torch
import functools
import argparse
import pdb
import traceback
from datetime import datetime

def generate(
    problem,
    task,
    total_samples,
    preprocess_success=False,
    preprocess_cost=False
):
    # 0. constant priors
    problem_dir = {
        "low": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/low/zero_shot_questions.json",
        "medium": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/medium/zero_shot_questions.json",
        "high": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/high/zero_shot_questions.json",
    }
    
    static_mapper = getattr(general_utils, f"{problem}_static_mapper")
    tunable_mapper = getattr(general_utils, f"{problem}_tunable_mapper")
    ranges = getattr(general_utils, "INFO")[task]
    _preprocessor = getattr(general_utils, f"{problem}_preprocessor")
    preprocessor = functools.partial(_preprocessor, preproc_success = preprocess_success, preproc_cost = preprocess_cost)
    
    all_samples = []
    failed_evaluations = []
    total_attempted = 0
    
    for level in problem_dir:
        verifier = Verifier(problem, task, level)
        
        # 1. load problems from the dir in problem_dir. Calculate the number of samples per problem per precision level. Note that for each problem, half of the samples are by random while half are by linear interpolation.
        with open(problem_dir[level], 'r') as f:
            problems = json.load(f)
        
        samples_per_problem = total_samples // (len(problems) * len(problem_dir))
        samples_random = samples_per_problem // 2
        samples_linear = samples_per_problem - samples_random
        
        # 2. loop for each problem, pass the entry to static_mapper for a mapped static tensor; sample with random and linear for the given amounts, and map the sampled tunable parameters using tunable_mapper
        for problem_entry in problems:
            profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{problem}/{problem_entry['profile']}.yaml"
            static_tensor = static_mapper(profile_path, level)
            qid = str(problem_entry['QID'])
            
            # Random sampling
            for _ in range(samples_random):
                sampled_params = copy.deepcopy(problem_entry["best_params"])
                if ranges["type"] == "int":
                    sampled_params[task] = np.random.randint(ranges["range"][problem][0], ranges["range"][problem][1] + 1)
                else:
                    sampled_params[task] = np.random.uniform(ranges["range"][problem][0], ranges["range"][problem][1])
                
                tunable_tensor = tunable_mapper(sampled_params)
                
                #pdb.set_trace()
                # 3. call verifier.metric to calculate result tensor
                total_attempted += 1
                try:
                    error, success, cost, _ = verifier.raw_metric(sampled_params, problem_entry['profile'], qid)
                    result_tensor = torch.tensor([error, cost])
                    
                    sample = (static_tensor, tunable_tensor, result_tensor)
                    all_samples.append(sample)
                except Exception as e:
                    failed_evaluations.append({
                        'level': level,
                        'qid': qid,
                        'profile': problem_entry['profile'],
                        'sampled_params': sampled_params,
                        'sampling_type': 'random',
                        'error_message': str(e),
                        'traceback': traceback.format_exc()
                    })
                    print(f"Failed evaluation (random sampling): QID {qid}, params {sampled_params}, error: {str(e)}")
                    continue
            
            # Linear interpolation sampling
            param_range = ranges["range"][problem]
            
            for i in range(samples_linear):
                alpha = i / max(1, samples_linear - 1)
                if ranges["type"] == "int":
                    sampled_value = int(param_range[0] + alpha * (param_range[1] - param_range[0]))
                else:
                    sampled_value = param_range[0] + alpha * (param_range[1] - param_range[0])
                
                sampled_params = copy.deepcopy(problem_entry["best_params"])
                sampled_params[task] = sampled_value
                tunable_tensor = tunable_mapper(sampled_params)
                #pdb.set_trace() 
                total_attempted += 1
                try:
                    error, success, cost, _ = verifier.raw_metric(sampled_params, problem_entry['profile'], qid)
                    result_tensor = torch.tensor([error, cost])
                    
                    sample = (static_tensor, tunable_tensor, result_tensor)
                    all_samples.append(sample)
                except Exception as e:
                    failed_evaluations.append({
                        'level': level,
                        'qid': qid,
                        'profile': problem_entry['profile'],
                        'sampled_params': sampled_params,
                        'sampling_type': 'linear_interpolation',
                        'error_message': str(e),
                        'traceback': traceback.format_exc()
                    })
                    print(f"Failed evaluation (linear sampling): QID {qid}, params {sampled_params}, error: {str(e)}")
                    continue
    
    # 4. Put all samples in all precision levels together, to train-test split, and save it in the given dir.
    save_dir = f"/home/ubuntu/dev/data/numerical/{problem}/{task}_{str(total_samples)}_preprocess-{str(preprocess_success)}-{str(preprocess_cost)}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Apply preprocessing to all samples
    processed_samples = []
    for entry in all_samples:
        static_tensor, tunable_tensor, result_tensor = entry
        if preprocess_success or preprocess_cost:
            processed_result = preprocessor(tunable_tensor, result_tensor)
            processed_samples.append((static_tensor, tunable_tensor, processed_result))
        else:
            processed_samples.append(entry)
    
    #pdb.set_trace()
    # Separate into input and output tensors
    static_tensors = torch.stack([sample[0] for sample in processed_samples])
    tunable_tensors = torch.stack([sample[1] for sample in processed_samples])
    result_tensors = torch.stack([sample[2] for sample in processed_samples])
    
    # Combine input features
    input_tensors = torch.cat([static_tensors, tunable_tensors], dim=1)
    
    # Train-test split
    n_samples = len(processed_samples)
    indices = torch.randperm(n_samples)
    
    train_end = int(0.8 * n_samples)
    
    train_indices = indices[:train_end]
    test_indices = indices[train_end:]
    
    # Create train and test sets as lists of tuples
    train_data = [(static_tensors[i], tunable_tensors[i], result_tensors[i]) for i in train_indices]
    test_data = [(static_tensors[i], tunable_tensors[i], result_tensors[i]) for i in test_indices]
    
    # Save datasets
    torch.save(train_data, os.path.join(save_dir, 'train.pth'))
    torch.save(test_data, os.path.join(save_dir, 'test.pth'))
    
    # Save metadata
    metadata = {
        'problem': problem,
        'task': task,
        'total_samples': len(all_samples),
        'preprocess_success': preprocess_success,
        'preprocess_cost': preprocess_cost,
        'static_dim': static_tensors.shape[1],
        'tunable_dim': tunable_tensors.shape[1],
        'result_dim': result_tensors.shape[1],
        'train_size': len(train_data),
        'test_size': len(test_data)
    }
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Log failure statistics
    failure_rate = len(failed_evaluations) / total_attempted if total_attempted > 0 else 0
    
    failure_log = {
        'timestamp': datetime.now().isoformat(),
        'problem': problem,
        'task': task,
        'total_attempted': total_attempted,
        'successful_samples': len(all_samples),
        'failed_evaluations_count': len(failed_evaluations),
        'failure_rate': failure_rate,
        'failed_evaluations': failed_evaluations
    }
    
    # Save failure log to data_gen directory
    log_filename = f"failure_log_{problem}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join("/home/ubuntu/dev/src/data_gen", log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(failure_log, f, indent=2, default=str)
    
    print(f"Dataset generated and saved to: {save_dir}")
    print(f"Total attempted: {total_attempted}")
    print(f"Successful samples: {len(all_samples)}")
    print(f"Failed evaluations: {len(failed_evaluations)} ({failure_rate:.2%})")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"Failure log saved to: {log_path}")
    
    return save_dir

def main():
    parser = argparse.ArgumentParser(description='Generate training data for numerical optimization problems')
    parser.add_argument('--problem', type=str, required=True, 
                       choices=['heat_1d', 'burgers_1d', 'euler_1d'],
                       help='Problem type to generate data for')
    parser.add_argument('--task', type=str, required=True,
                       choices=['cfl', 'n_space', 'beta', 'k'],
                       help='Task/parameter to optimize')
    parser.add_argument('--total_samples', type=int, required=True,
                       help='Total number of samples to generate')
    parser.add_argument('--preprocess_success', action='store_true',
                       help='Apply preprocessing to success values')
    parser.add_argument('--preprocess_cost', action='store_true', 
                       help='Apply preprocessing to cost values')
    
    args = parser.parse_args()
    
    print(f"Generating data for {args.problem} - {args.task}")
    print(f"Total samples: {args.total_samples}")
    print(f"Preprocess success: {args.preprocess_success}")
    print(f"Preprocess cost: {args.preprocess_cost}")
    
    save_dir = generate(
        problem=args.problem,
        task=args.task, 
        total_samples=args.total_samples,
        preprocess_success=args.preprocess_success,
        preprocess_cost=args.preprocess_cost
    )
    
    print(f"Data generation complete. Saved to: {save_dir}")

if __name__ == "__main__":
    main()