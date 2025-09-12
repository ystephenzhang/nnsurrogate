import sys
sys.path.append("/home/ubuntu/dev/src")
from general_utils import Verifier
import general_utils
import json
import numpy as np
import os
import copy
import glob
import torch
import functools
import argparse
import pdb
import traceback
from datetime import datetime
from tqdm import tqdm

def save_incremental_checkpoint(all_samples, save_dir, checkpoint_filename):
    """
    Save incremental checkpoint of samples during generation.
    
    Args:
        all_samples: List of samples generated so far
        save_dir: Base save directory
        checkpoint_filename: Name for checkpoint file
    """
    try:
        # Create checkpoints directory
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save current samples
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(all_samples, checkpoint_path)
        
        # Also save a quick metadata file
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'samples_count': len(all_samples),
            'checkpoint_file': checkpoint_filename
        }
        
        metadata_path = os.path.join(checkpoint_dir, f"metadata_{len(all_samples)}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"🔄 Saved checkpoint: {len(all_samples)} samples to {checkpoint_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint at {len(all_samples)} samples: {e}")

def cleanup_checkpoints(save_dir, keep_final=False):
    """
    Clean up checkpoint files after successful completion.
    
    Args:
        save_dir: Base save directory
        keep_final: Whether to keep the final checkpoint file
    """
    try:
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            import shutil
            if keep_final:
                # Keep only the largest checkpoint (final one)
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
                if checkpoint_files:
                    # Find the checkpoint with the highest sample count
                    def extract_sample_count(filename):
                        import re
                        match = re.search(r'checkpoint_(\d+)_samples\.pt', os.path.basename(filename))
                        return int(match.group(1)) if match else 0
                    
                    checkpoint_files.sort(key=extract_sample_count)
                    final_checkpoint = checkpoint_files[-1]
                    
                    # Remove all but the final checkpoint
                    for checkpoint_file in checkpoint_files[:-1]:
                        os.remove(checkpoint_file)
                        # Remove corresponding metadata
                        sample_count = extract_sample_count(checkpoint_file)
                        metadata_file = os.path.join(checkpoint_dir, f"metadata_{sample_count}.json")
                        if os.path.exists(metadata_file):
                            os.remove(metadata_file)
                    
                    print(f"🧹 Cleaned up intermediate checkpoints, kept final: {os.path.basename(final_checkpoint)}")
            else:
                # Remove entire checkpoints directory
                shutil.rmtree(checkpoint_dir)
                print(f"🧹 Removed all checkpoint files")
    except Exception as e:
        print(f"⚠️  Warning: Failed to cleanup checkpoints: {e}")

def generate(
    problem,
    task,
    total_samples=None,
    save_dir=None,
    preprocess_cost=False,
    samples_per_problem=None
):
    # Set random seeds for reproducible results
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 0. constant priors
    problem_dir = {
        "low": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/low/zero_shot_questions.json",
        "medium": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/medium/zero_shot_questions.json",
        "high": f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/high/zero_shot_questions.json",
    }
    
    static_mapper = getattr(general_utils, f"{problem}_static_mapper")
    tunable_mapper = getattr(general_utils, f"{problem}_tunable_mapper")
    ranges = getattr(general_utils, "INFO")[task]
    #_preprocessor = getattr(general_utils, f"{problem}_preprocessor")
    #preprocessor = functools.partial(_preprocessor, preproc_cost = preprocess_cost)
    
    all_samples = []
    failed_evaluations = []
    total_attempted = 0
    
    for level in problem_dir:
        verifier = Verifier(problem, task, level)
        
        # 1. load problems from the dir in problem_dir. Calculate the number of samples per problem per precision level. Note that for each problem, half of the samples are by random while half are by linear interpolation.
        with open(problem_dir[level], 'r') as f:
            problems = json.load(f)
        
        if samples_per_problem is not None:
            # Use samples_per_problem: m linear + m random per problem
            samples_linear = samples_per_problem
            samples_random = samples_per_problem
        else:
            # Use total_samples divided across all problems
            samples_per_problem_total = total_samples // (len(problems) * len(problem_dir))
            samples_random = samples_per_problem_total // 2
            samples_linear = samples_per_problem_total - samples_random
        
        # 2. loop for each problem, pass the entry to static_mapper for a mapped static tensor; sample with random and linear for the given amounts, and map the sampled tunable parameters using tunable_mapper
        for problem_entry in problems:
            profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{problem}/{problem_entry['profile']}.yaml"
            static_tensor = static_mapper(profile_path, level)
            qid = str(problem_entry['QID'])
            
            # Random sampling
            for _ in tqdm(range(samples_random), desc="Random Sampling"):
                sampled_params = copy.deepcopy(problem_entry["best_params"])
                sampled_params[task] = np.random.randint(ranges["range"][0], ranges["range"][1] + 1)
                
                # Special handling for ns_channel_2d mesh_x sampling
                if problem == "ns_channel_2d" and task == "mesh_x":
                    aspect_ratio = problem_entry.get("aspect_ratio", 0.25)  # Default to 4.0 if not found
                    sampled_params["mesh_y"] = min(int(sampled_params["mesh_x"] * aspect_ratio), 64)
                
                if problem == "ns_transient_2d" and task == "resolution":
                    sampled_params[task] = 2 * (sampled_params[task] // 2)
                
                tunable_tensor = tunable_mapper(sampled_params)
                
                #pdb.set_trace()
                # 3. call verifier.metric to calculate result tensor
                total_attempted += 1
                try:
                    error, success, cost, _ = verifier.raw_metric(sampled_params, problem_entry['profile'], qid)
                    result_tensor = torch.tensor([success, error, cost])
                    
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
                    
                # Save incremental checkpoint every 50 samples
                if len(all_samples) % 50 == 0:
                    save_incremental_checkpoint(all_samples, save_dir, f"checkpoint_{len(all_samples)}_samples.pt")
            
            # Linear interpolation sampling
            param_range = ranges["range"]
            
            for i in tqdm(range(samples_linear), desc="Linear sampling"):
                alpha = i / max(1, samples_linear - 1)
                sampled_value = int(param_range[0] + alpha * (param_range[1] - param_range[0]))
                
                sampled_params = copy.deepcopy(problem_entry["best_params"])
                sampled_params[task] = sampled_value
                
                # Special handling for ns_channel_2d mesh_x sampling
                if problem == "ns_channel_2d" and task == "mesh_x":
                    aspect_ratio = problem_entry.get("aspect_ratio", 0.25)  # Default to 4.0 if not found
                    sampled_params["mesh_y"] = min(int(sampled_params["mesh_x"] * aspect_ratio), 64)
                
                if problem == "ns_transient_2d" and task == "resolution":
                    sampled_params[task] = 2 * (sampled_params[task] // 2)
                
                tunable_tensor = tunable_mapper(sampled_params)
                #pdb.set_trace() 
                total_attempted += 1
                try:
                    error, success, cost, _ = verifier.raw_metric(sampled_params, problem_entry['profile'], qid)
                    result_tensor = torch.tensor([success, error, cost])
                    
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
                
                # Save incremental checkpoint every 50 samples
                if len(all_samples) % 50 == 0:
                    save_incremental_checkpoint(all_samples, save_dir, f"checkpoint_{len(all_samples)}_samples.pt")
    
    # 4. Put all samples in all precision levels together, to train-test split, and save it in the given dir.
    #save_dir = f"/home/ubuntu/dev/data/numerical/{problem}/{task}_{str(total_samples)}_p{str(preprocess_cost)}"
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    
    # Apply preprocessing to all samples
    processed_samples = []
    for entry in all_samples:
        static_tensor, tunable_tensor, result_tensor = entry
        if preprocess_cost:
            processed_result = preprocessor(tunable_tensor, result_tensor)
            processed_samples.append((static_tensor, tunable_tensor, processed_result))
        else:
            processed_samples.append(entry)
    
    #pdb.set_trace()
    # Separate into input and output tensors
    static_tensors = torch.stack([sample[0] for sample in processed_samples])
    tunable_tensors = torch.stack([sample[1] for sample in processed_samples])
    result_tensors = torch.stack([sample[2] for sample in processed_samples])
    
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
    torch.save(train_data, os.path.join(save_dir, 'train', 'train.pt'))
    torch.save(test_data, os.path.join(save_dir, 'test', 'test.pt'))
    
    # Save metadata
    metadata = {
        'problem': problem,
        'task': task,
        'total_samples': len(all_samples),
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
    
    # Clean up checkpoint files after successful completion
    cleanup_checkpoints(save_dir, keep_final=False)
    
    return save_dir

def complementary_sample(
    problem,
    task,
    precision_level,
    num_samples,
    existing_data_path,
    sampling_range=None,
    seed_offset=1000
):
    """
    Generate complementary random samples and append to existing dataset.
    
    Args:
        problem: Problem type (e.g., 'heat_1d', 'euler_1d')
        task: Task parameter to optimize (e.g., 'n_space', 'cfl')
        precision_level: Precision level ('low', 'medium', 'high')
        num_samples: Number of additional samples to generate
        existing_data_path: Path to existing dataset directory
        sampling_range: Optional custom range [min, max], uses default if None
        seed_offset: Offset to add to base seed to avoid duplicating samples from complete generation
    """
    # Load existing metadata to get dataset size for more unique seeding
    metadata_path = os.path.join(existing_data_path, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create a more unique seed based on existing dataset size and offset
    existing_total_samples = metadata.get('total_samples', 0)
    complementary_seed = 42 + seed_offset + existing_total_samples
    
    # Set different random seeds to avoid duplicating samples from complete generation
    np.random.seed(complementary_seed)
    torch.manual_seed(complementary_seed)
    
    print(f"Using complementary random seed: {complementary_seed} (base: 42 + offset: {seed_offset} + existing_samples: {existing_total_samples})")
    
    # Get utility functions and ranges
    static_mapper = getattr(general_utils, f"{problem}_static_mapper")
    tunable_mapper = getattr(general_utils, f"{problem}_tunable_mapper") 
    ranges = getattr(general_utils, "INFO")[task]
    
    # Use custom range if provided, otherwise use default
    if sampling_range is None:
        param_range = ranges["range"]
    else:
        param_range = sampling_range
    
    #preprocess_cost = metadata.get('preprocess_cost', False)
    #_preprocessor = getattr(general_utils, f"{problem}_preprocessor")
    #preprocessor = functools.partial(_preprocessor, preproc_cost=preprocess_cost)
    
    # Load problem definitions for the specified precision level
    problem_dir = f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/{precision_level}/zero_shot_questions.json"
    with open(problem_dir, 'r') as f:
        problems = json.load(f)
    
    # Initialize verifier
    verifier = Verifier(problem, task, precision_level)
    
    # Generate new samples
    new_samples = []
    failed_evaluations = []
    total_attempted = 0
    
    # Distribute samples across all problems in the precision level
    samples_per_problem = num_samples // len(problems)
    remaining_samples = num_samples % len(problems)
    
    for prob_idx, problem_entry in enumerate(problems):
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{problem}/{problem_entry['profile']}.yaml"
        static_tensor = static_mapper(profile_path, precision_level)
        qid = str(problem_entry['QID'])
        
        # Determine number of samples for this problem
        current_samples = samples_per_problem
        if prob_idx < remaining_samples:
            current_samples += 1
        
        # Generate random samples for this problem
        for _ in tqdm(range(current_samples), desc=f"Complementary sampling QID {qid}"):
            sampled_params = copy.deepcopy(problem_entry["best_params"])
            sampled_params[task] = np.random.randint(param_range[0], param_range[1] + 1)
            
            # Special handling for ns_channel_2d mesh_x sampling
            if problem == "ns_channel_2d" and task == "mesh_x":
                aspect_ratio = problem_entry.get("aspect_ratio", 0.25)
                sampled_params["mesh_y"] = min(int(sampled_params["mesh_x"] * aspect_ratio), 64)
            
            tunable_tensor = tunable_mapper(sampled_params)
            
            total_attempted += 1
            try:
                error, success, cost, _ = verifier.raw_metric(sampled_params, problem_entry['profile'], qid)
                result_tensor = torch.tensor([success, error, cost])
                
                sample = (static_tensor, tunable_tensor, result_tensor)
                new_samples.append(sample)
            except Exception as e:
                failed_evaluations.append({
                    'level': precision_level,
                    'qid': qid,
                    'profile': problem_entry['profile'],
                    'sampled_params': sampled_params,
                    'sampling_type': 'complementary_random',
                    'error_message': str(e),
                    'traceback': traceback.format_exc()
                })
                # Only print error if it's not the common simulation file missing error
                if "can't open file" not in str(e):
                    print(f"Failed evaluation (complementary sampling): QID {qid}, params {sampled_params}, error: {str(e)}")
                continue
    
    if not new_samples:
        print("No new samples generated successfully!")
        return
    
    # Apply preprocessing to new samples
    processed_new_samples = []
    for entry in new_samples:
        static_tensor, tunable_tensor, result_tensor = entry
        if preprocess_cost:
            processed_result = preprocessor(tunable_tensor, result_tensor)
            processed_new_samples.append((static_tensor, tunable_tensor, processed_result))
        else:
            processed_new_samples.append(entry)
    
    # Load existing train and test data
    train_path = os.path.join(existing_data_path, 'train', 'train.pt')
    test_path = os.path.join(existing_data_path, 'test', 'test.pt')
    
    existing_train_data = torch.load(train_path)
    existing_test_data = torch.load(test_path)
    
    # Split new samples maintaining the same train/test ratio as existing data
    existing_total = len(existing_train_data) + len(existing_test_data)
    existing_train_ratio = len(existing_train_data) / existing_total
    
    n_new_samples = len(processed_new_samples)
    n_new_train = int(n_new_samples * existing_train_ratio)
    
    # Randomly shuffle new samples and split
    indices = torch.randperm(n_new_samples)
    train_indices = indices[:n_new_train]
    test_indices = indices[n_new_train:]
    
    new_train_data = [processed_new_samples[i] for i in train_indices]
    new_test_data = [processed_new_samples[i] for i in test_indices]
    
    # Combine with existing data
    combined_train_data = existing_train_data + new_train_data
    combined_test_data = existing_test_data + new_test_data
    
    # Save updated datasets
    torch.save(combined_train_data, train_path)
    torch.save(combined_test_data, test_path)
    
    # Update metadata
    metadata['total_samples'] += len(new_samples)
    metadata['train_size'] = len(combined_train_data)
    metadata['test_size'] = len(combined_test_data)
    metadata['complementary_samples_added'] = len(new_samples)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Log complementary sampling results
    failure_rate = len(failed_evaluations) / total_attempted if total_attempted > 0 else 0
    
    failure_log = {
        'timestamp': datetime.now().isoformat(),
        'sampling_type': 'complementary',
        'problem': problem,
        'task': task,
        'precision_level': precision_level,
        'existing_data_path': existing_data_path,
        'total_attempted': total_attempted,
        'successful_samples': len(new_samples),
        'failed_evaluations_count': len(failed_evaluations),
        'failure_rate': failure_rate,
        'failed_evaluations': failed_evaluations
    }
    
    # Save failure log
    log_filename = f"complementary_failure_log_{problem}_{task}_{precision_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join("/home/ubuntu/dev/src/data_gen", log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(failure_log, f, indent=2, default=str)
    
    print(f"Complementary sampling complete!")
    print(f"Added {len(new_samples)} new samples to: {existing_data_path}")
    print(f"Total attempted: {total_attempted}")
    print(f"Failed evaluations: {len(failed_evaluations)} ({failure_rate:.2%})")
    print(f"Updated train size: {len(combined_train_data)}")
    print(f"Updated test size: {len(combined_test_data)}")
    print(f"Failure log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate training data for numerical optimization problems')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Complete sampling command
    complete_parser = subparsers.add_parser('complete', help='Generate complete dataset from scratch')
    complete_parser.add_argument('-p', '--problem', type=str, required=True,
                       help='Problem type to generate data for')
    complete_parser.add_argument('-t', '--task', type=str, required=True,
                       help='Task/parameter to optimize')
    complete_parser.add_argument('-n', '--total_samples', type=int,
                       help='Total number of samples to generate')
    complete_parser.add_argument('-m', '--samples_per_problem', type=int,
                       help='Number of samples per problem (m linear + m random each)')
    complete_parser.add_argument('-s', '--save_dir', type=str,
                       help='Directory to save the generated dataset')
    complete_parser.add_argument('--preprocess_cost', action='store_true', 
                       help='Apply preprocessing to cost values')
    
    # Complementary sampling command
    comp_parser = subparsers.add_parser('complementary', help='Add complementary samples to existing dataset')
    comp_parser.add_argument('-p', '--problem', type=str, required=True,
                       help='Problem type to generate data for')
    comp_parser.add_argument('-t', '--task', type=str, required=True,
                       help='Task/parameter to optimize')
    comp_parser.add_argument('-l', '--precision_level', type=str, required=True,
                       choices=['low', 'medium', 'high'],
                       help='Precision level for sampling')
    comp_parser.add_argument('-n', '--num_samples', type=int, required=True,
                       help='Number of complementary samples to add')
    comp_parser.add_argument('-d', '--existing_data_path', type=str, required=True,
                       help='Path to existing dataset directory')
    comp_parser.add_argument('-r', '--sampling_range', nargs=2, type=int,
                       help='Custom sampling range [min, max] (optional)')
    comp_parser.add_argument('-s', '--seed_offset', type=int, default=1000,
                       help='Seed offset to avoid duplicate samples (default: 1000)')
    
    args = parser.parse_args()
    
    if args.command == 'complete':
        # Validation: either total_samples or samples_per_problem must be provided
        if not args.total_samples and not args.samples_per_problem:
            parser.error("Either --total_samples or --samples_per_problem must be specified")
        if args.total_samples and args.samples_per_problem:
            parser.error("Cannot specify both --total_samples and --samples_per_problem")
        
        # Auto-generate save_dir if not provided
        if not args.save_dir:
            if args.samples_per_problem:
                args.save_dir = f"/home/ubuntu/dev/data/numerical/{args.problem}/{args.task}_m{args.samples_per_problem}_p{str(args.preprocess_cost)}"
            else:
                args.save_dir = f"/home/ubuntu/dev/data/numerical/{args.problem}/{args.task}_{args.total_samples}_p{str(args.preprocess_cost)}"
        
        print(f"Generating complete dataset for {args.problem} - {args.task}")
        if args.samples_per_problem:
            print(f"Samples per problem: {args.samples_per_problem} linear + {args.samples_per_problem} random")
        else:
            print(f"Total samples: {args.total_samples}")
        print(f"Preprocess cost: {args.preprocess_cost}")
        print(f"Save directory: {args.save_dir}")
        
        save_dir = generate(
            problem=args.problem,
            task=args.task, 
            total_samples=args.total_samples,
            save_dir=args.save_dir,
            preprocess_cost=args.preprocess_cost,
            samples_per_problem=args.samples_per_problem
        )
        
        print(f"Data generation complete. Saved to: {save_dir}")
    
    elif args.command == 'complementary':
        print(f"Adding complementary samples for {args.problem} - {args.task}")
        print(f"Precision level: {args.precision_level}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Existing data path: {args.existing_data_path}")
        if args.sampling_range:
            print(f"Custom sampling range: {args.sampling_range}")
        
        complementary_sample(
            problem=args.problem,
            task=args.task,
            precision_level=args.precision_level,
            num_samples=args.num_samples,
            existing_data_path=args.existing_data_path,
            sampling_range=args.sampling_range,
            seed_offset=args.seed_offset
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()