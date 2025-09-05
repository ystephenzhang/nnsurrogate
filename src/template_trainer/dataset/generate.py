import os, glob, sys
sys.path.append("/home/ubuntu/dev/src")
import pdb
from general_utils import *
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import json

precision_levels = {
    "euler_1d":
        {
            "high":0.01,
            "medium":0.02,
            "low":0.08
        }
}
def get_profile_from_fixed_params(problem, task, precision, fixed_params, target_parameter):
    """
    Get profile number based on problem, task, precision, and fixed parameters.
    
    Args:
        problem (str): Problem name (e.g., "euler_1d")
        task (str): Task name (e.g., "cfl", "n_space") 
        precision (str): Precision level (e.g., "high", "medium", "low")
        fixed_params (dict): Dictionary of fixed parameters and their values
        target_parameter (str): The parameter being optimized
    
    Returns:
        str: Profile name (e.g., "p1", "p2", "p3")
    """
    import json
    import os
    
    # Construct path to zero_shot_questions.json
    questions_path = f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/{precision}/zero_shot_questions.json"
    
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
    
    # Find matching entry by comparing fixed parameters
    for entry in questions_data:
        if entry['target_parameter'] != target_parameter:
            continue
            
        # Extract fixed parameters from best_params (excluding target parameter)
        entry_fixed_params = {k: v for k, v in entry['best_params'].items() 
                             if k != target_parameter}
        
        # Check if fixed parameters match
        if entry_fixed_params == fixed_params:
            return entry['profile']
    
    # If no exact match found, try to infer from case mapping (for euler_1d)
    if problem == "euler_1d":
        # Common case to profile mapping for euler_1d
        case_to_profile = {"sod": "p1", "lax": "p2", "mach_3": "p3"}
        
        # Try to find a case that matches the fixed parameters structure
        for entry in questions_data:
            if entry['target_parameter'] == target_parameter:
                entry_fixed_params = {k: v for k, v in entry['best_params'].items() 
                                     if k != target_parameter}
                # If the structure matches (same keys), use the case-based mapping
                if set(entry_fixed_params.keys()) == set(fixed_params.keys()):
                    # Find the case and return corresponding profile
                    if 'case' in entry:
                        return case_to_profile.get(entry['case'], "p1")
    
    # Default fallback
    return "p1"

def _x_s_to_profile(problem, x_s):
    if problem == "euler_1d":
        if x_s[0] == 1:
            return 'p1'
        elif x_s[1] == 1:
            return 'p2'
        elif x_s[2] == 1:
            return 'p3'

class SimRewardGenerator():
    def __init__(self, cfg, problem):
        self.problem = problem
        self.cfg = cfg
        self.data_samples = []  # Store (x_s, x_t, y) tuples
    
    def generate_from_existing(self,
                               tensor_path,
                                change_precision=None,
                               original_perc="high"
                               ):
        data = torch.load(tensor_path)
        verifier = Verifier(
            self.problem,
            self.cfg.tasks[0],
            change_precision
        )
        for entry in data:
            x_s, x_t, y = entry
            if change_precision:
                x_s[-1] = precision_levels[self.problem][change_precision]
            y[0], _, _ = verifier.metric({
                "cfl": torch.round(x_t[0] * 1000) / 1000,
                "beta":x_t[1],
                "k": x_t[2],
                "n_space": int(x_t[3])
            }, _x_s_to_profile(self.problem, x_s), True)
            
        output_path = tensor_path.replace(original_perc, change_precision)
        torch.save(data, output_path) 
                
         
    def calc_reward(self, verifier, params, profile, qid):
        # For burgers_1d, generate synthetic rewards to avoid running actual simulations
        if self.problem == "burgers_1d":
            # Generate realistic synthetic success and cost values
            import numpy as np
            cfl = params.get('cfl', 0.25)
            # Better CFL values (smaller) tend to have higher success rates
            success_rate = min(1.0, 2.0 / (cfl + 0.01))  # Higher success for smaller CFL
            success = success_rate + np.random.normal(0, 0.1)  # Add some noise
            success = max(0.0, min(1.0, success))  # Clamp to [0,1]
            
            # Cost increases with smaller CFL (more time steps needed)
            base_cost = 1000000 / cfl  # Rough inverse relationship
            cost = int(base_cost * (1 + np.random.normal(0, 0.1)))  # Add noise
            cost = max(10000, cost)  # Minimum cost
            
            return success, cost
        else:
            # For other problems, use normal verification
            success, cost, _ = verifier.metric(params, profile, str(qid))
            return success, cost
    
    def _get_precision_level_value(self, cfg, precision_level):
        '''
        Extract the float tolerance value for a given precision level from checkout config.
        
        Args:
            cfg: Checkout configuration dictionary
            precision_level (str): 'low', 'medium', or 'high'
            
        Returns:
            float: The tolerance value for the specified precision level
        '''
        precision_levels = cfg.get('precision_levels', {})
        
        if not precision_levels:
            raise ValueError("No precision levels found in checkout config")
        
        if precision_level not in precision_levels:
            raise ValueError(f"Precision level '{precision_level}' not found. Available: {list(precision_levels.keys())}")
        
        level_config = precision_levels[precision_level]
        
        # Extract the first tolerance value found
        for key, value in level_config.items():
            if 'tolerance' in key:
                return float(value)
        
        raise ValueError(f"No tolerance value found for precision level '{precision_level}'")
    
    def generate_data(self,
                      precision_level,
                      sampling_method=None,
                      n_samples_per_combination=None):
        """
        Generate training data by linearly sampling tunable parameters and evaluating them.
        
        Args:
            precision_level (str): Designated precision level to use ('low', 'medium', or 'high').
            n_samples_per_combination (int): Number of samples per task per profile per non-target parameter combination.
                                           If None, uses config n_sample.
        
        For each active profile, task, and non-target parameter combination:
        1. Extract static parameters from YAML profile  
        2. Extract precision level float value from checkout config
        3. Linearly generate tunable parameters within their bounds
        4. Evaluate soft success and cost using the verifier
        5. Store as (x_s, x_t, y) tuples where:
           - x_s: static parameters + precision level value (tensor)
           - x_t: tunable parameters (tensor)
           - y: reward (soft_success, cost) (tensor)
        """
        # Load checkout configuration to get active profiles and task-specific parameters
        checkout_config = self._load_checkout_config()
        
        # Use config-specified profiles if provided, otherwise use checkout config active profiles
        if hasattr(self.cfg, 'profiles') and self.cfg.profiles is not None:
            active_profiles = self.cfg.profiles
            print(f"Using config-specified profiles: {active_profiles}")
        else:
            active_profiles = checkout_config.get('profiles', {}).get('active_profiles', [])
            print(f"Using checkout config active profiles: {active_profiles}")
        # Extract the float value for the specified precision level
        precision_level_value = self._get_precision_level_value(checkout_config, precision_level)
        
        # Use provided sample count or get from config
        if n_samples_per_combination is not None:
            n_sample = n_samples_per_combination
        else:
            n_sample = self.cfg.n_sample
        
        # active profiles
        selected_files = []
        for profile_name in active_profiles:
            profile_path = os.path.join(self.cfg.profile_path, f"{profile_name}.yaml")
            if os.path.exists(profile_path):
                selected_files.append(profile_path)
            else:
                print(f"Warning: Profile {profile_name}.yaml not found at {profile_path}")
        
        # non-target params
        from itertools import product
        non_target_params = {}
        for task in checkout_config.get("target_parameters", {}).keys():
            task_config = checkout_config["target_parameters"][task]
            non_target_config = task_config.get("non_target_parameters", {})
            
            # Get all parameter names and their possible values
            param_names = []
            param_values = []
            for param_name, param_value in non_target_config.items():
                param_names.append(param_name)
                if isinstance(param_value, list):
                    param_values.append(param_value)
                else:
                    param_values.append([param_value])
            
            # Generate all combinations using itertools.product
            combinations = []
            for combo in product(*param_values):
                combinations.append(tuple(zip(param_names, combo)))
            
            non_target_params[task] = combinations 
        
        total_combinations = len(selected_files) \
                                * sum([len(non_target_params[task]) for task in non_target_params]) \
                                * n_sample
        if not sampling_method:
            sampling_method = getattr(self.cfg, 'sampling_method', 'random')
        print(f"Generating {total_combinations} samples")
        print(f"Active profiles: {active_profiles}")
        print(f"Precision level: {precision_level} (value: {precision_level_value})")
        print(f"Sampling method: {sampling_method}")
        print(f"Samples per combination: {n_sample}")
        
        for profile_path in selected_files:
            profile_name = os.path.basename(profile_path).replace('.yaml', '')
            print(f"Processing profile: {profile_name}")
            
            # Extract static parameters from YAML
            x_s_dict = extract_static(profile_path)
            
            for task in self.cfg.tasks:
                print(f"  Processing task: {task}")
                
                # Get non-target parameter combinations for this task
                task_non_targets = non_target_params.get(task, [])
                
                for non_targets_tuple in task_non_targets:
                    # Convert tuple of (param_name, param_value) pairs to dictionary
                    fixed_params = dict(non_targets_tuple)
                    print(f"    Processing non-target params: {fixed_params}")
                
                    # Add precision level value to static parameters
                    x_s_dict_with_precision = x_s_dict.copy()
                    if self.cfg.add_precision:
                        x_s_dict_with_precision['precision_level'] = precision_level_value
                    
                    # Initialize verifier for this problem/task combination
                    ver = Verifier(self.problem,
                                   task,
                                   precision_level)
                    
                    # Generate samples for tunable parameters based on sampling method
                    if sampling_method == 'linear':
                        x_t_samples = self._generate_linear_tunable_samples(
                            self.problem, task, n_sample, fixed_params
                        )
                    elif sampling_method == 'random':
                        x_t_samples = self._generate_random_tunable_samples(
                            self.problem, task, n_sample, fixed_params
                        )
                    else:
                        raise ValueError(f"Unknown sampling method: {sampling_method}. Use 'linear' or 'random'.")
                    
                    # Collect y_tensors for this batch
                    batch_y_tensors = []
                    


                    for i, x_t_dict in enumerate(x_t_samples):
                        #try:
                        # Calculate soft success and cost
                        success, cost = self.calc_reward(ver, x_t_dict, profile_name)
                        
                        # Convert success to soft success (already handled by calc_reward)
                        soft_success = success
                        
                        # Convert to tensors
                        x_s_tensor = self._dict_to_tensor(x_s_dict_with_precision)
                        x_t_tensor = self._dict_to_tensor(x_t_dict)
                        y_tensor = torch.tensor([soft_success, cost], dtype=torch.float32)
                        
                        # Apply problem-specific preprocessing
                        if self.problem == "euler_1d":
                            # Step 1: Remove first two dimensions from x_s_tensor (static parameters)
                            if x_s_tensor.shape[0] >= 2:
                                x_s_tensor = x_s_tensor[2:]
                            
                            # Step 2: Check for NaN in any tensor and skip if found
                            has_nan = (torch.isnan(x_s_tensor).any() or 
                                      torch.isnan(x_t_tensor).any() or 
                                      torch.isnan(y_tensor).any())
                            
                            if has_nan:
                                print(f"      Skipping sample {i} due to NaN values")
                                continue
                        
                        elif self.problem == "burgers_1d":
                            # For burgers_1d: x_s_tensor should only contain case one-hot encoding (5 dims)
                            # No need to remove dimensions - it should already be correct
                            
                            # Check for NaN in any tensor and skip if found
                            has_nan = (torch.isnan(x_s_tensor).any() or 
                                      torch.isnan(x_t_tensor).any() or 
                                      torch.isnan(y_tensor).any())
                            
                            if has_nan:
                                print(f"      Skipping sample {i} due to NaN values")
                                continue
                        
                        # Store for batch statistics
                        batch_y_tensors.append(y_tensor)
                        
                        # Store the sample
                        self.data_samples.append((x_s_tensor, x_t_tensor, y_tensor))
                        if (i + 1) % 10 == 0:
                            print(f"      Generated {i + 1}/{n_sample} samples for this combination")
                    
                    # Log batch statistics
                    if batch_y_tensors:
                        batch_tensor = torch.stack(batch_y_tensors)
                        y0_values = batch_tensor[:, 0]  # soft_success values
                        y1_values = batch_tensor[:, 1]  # cost values
                        
                        with open('/home/ubuntu/dev/src/outputs/log/y_tensor_stats.log', 'a') as f:
                            f.write(f"Batch (rofile={profile_name}): "
                                   f"y[0] mean={y0_values.mean().item():.6f}, range=({y0_values.min().item():.6f}, {y0_values.max().item():.6f}), "
                                   f"y[1] mean={y1_values.mean().item():.6f}, range=({y1_values.min().item():.6f}, {y1_values.max().item():.6f})\n")
                        
                        #except Exception as e:
                        #    print(f"      Error evaluating sample {i}: {e}")
                        #    continue
        
        print(f"Generated {len(self.data_samples)} total samples")
        self._save_data(precision_level)
    
    def generate_data_from_dataset(self,
                                   problem,
                                   task,
                                   precision_level,
                                   mode='zero_shot',
                                   sampling_method=None,
                                   n_samples_per_entry=None):
        """
        Generate training data by sampling directly from zero_shot_questions.json dataset entries.
        
        Args:
            problem (str): Problem name (e.g., "euler_1d")
            task (str): Task name (e.g., "cfl", "n_space")
            precision_level (str): Precision level ('low', 'medium', 'high')
            mode (str): Either 'zero_shot' or 'iterative' 
            sampling_method (str): 'random' or 'linear' (from config if None)
            n_samples_per_entry (int): Number of samples per dataset entry (from config if None)
        
        For each entry in the dataset:
        1. Extract profile and fixed parameters from best_params
        2. Use the same sampling logic as original generate_data
        3. Generate n_samples_per_entry for each entry
        4. Store as (x_s, x_t, y) tuples
        """
        # Get parameters from config
        if n_samples_per_entry is None:
            n_samples_per_entry = getattr(self.cfg, 'n_sample', 30)
        
        if sampling_method is None:
            sampling_method = getattr(self.cfg, 'sampling_method', 'random')
            if isinstance(sampling_method, list):
                sampling_method = sampling_method[0]  # Use first method if list
        
        # Construct path to questions file
        questions_path = f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/{task}/{precision_level}/{mode}_questions.json"
        
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        
        # Load questions dataset
        print(f"Loading dataset from: {questions_path}")
        with open(questions_path, 'r') as f:
            dataset_entries = json.load(f)
        
        print(f"Found {len(dataset_entries)} entries in dataset")
        print(f"Generating {n_samples_per_entry} samples per entry ({n_samples_per_entry * len(dataset_entries)} total)")
        print(f"Using sampling method: {sampling_method}")
        
        # Get precision level value
        precision_level_value = precision_levels[problem][precision_level]
        
        # Process each dataset entry
        total_samples = 0
        for entry_idx, entry in enumerate(dataset_entries):
            profile = entry['profile']
            qid = entry['QID']
            target_parameter = entry['target_parameter']
            best_params = entry['best_params']
            
            print(f"Processing entry {entry_idx + 1}/{len(dataset_entries)}: {profile}, QID: {qid}")
            
            # Load profile configuration
            profile_path = os.path.join(getattr(self.cfg, 'profile_path', 
                                              f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{problem}"),
                                       f"{profile}.yaml")
            
            if not os.path.exists(profile_path):
                print(f"Warning: Profile {profile}.yaml not found at {profile_path}, skipping")
                continue
            
            with open(profile_path, 'r') as f:
                profile_config = yaml.safe_load(f)
            
            # Create x_s (static parameters) from profile config + precision
            x_s_dict = profile_config.copy()
            if getattr(self.cfg, 'add_precision', True):
                x_s_dict['precision_level'] = precision_level_value
            
            x_s_dict_with_precision = x_s_dict.copy()
            
            # Extract fixed parameters from best_params (excluding target parameter)
            fixed_params = {k: v for k, v in best_params.items() if k != target_parameter}
            
            print(f"  Fixed parameters: {fixed_params}")
            print(f"  Target parameter: {target_parameter}")
            
            # Initialize verifier for this profile
            ver = Verifier(problem, task, precision_level)
            
            # Generate samples for this entry
            if sampling_method == "linear":
                x_t_samples = self._generate_linear_tunable_samples(
                    problem, task, n_samples_per_entry, fixed_params
                )
            elif sampling_method == "random":
                x_t_samples = self._generate_random_tunable_samples(
                    problem, task, n_samples_per_entry, fixed_params
                )
            else:
                raise ValueError(f"Unknown sampling method: {sampling_method}. Use 'linear' or 'random'.")
            
            # Collect y_tensors for this entry
            entry_y_tensors = []
            
            for i, x_t_dict in enumerate(x_t_samples):
                # Calculate soft success and cost
                success, cost = self.calc_reward(ver, x_t_dict, profile, qid)
                
                # Convert success to soft success (already handled by calc_reward)
                soft_success = success
                
                # Convert to tensors
                x_s_tensor = self._dict_to_tensor(x_s_dict_with_precision)
                x_s_tensor = x_s_tensor[2:]
                
                x_t_tensor = self._dict_to_tensor(x_t_dict)

                y_tensor = torch.tensor([soft_success, cost], dtype=torch.float32)
                #pdb.set_trace() 
                # Check for NaN values and skip if found
                has_nan = (torch.isnan(x_s_tensor).any() or 
                          torch.isnan(x_t_tensor).any() or 
                          torch.isnan(y_tensor).any())
                
                if has_nan:
                    print(f"    Skipping sample {i} due to NaN values")
                    continue
                
                # Store for entry statistics
                entry_y_tensors.append(y_tensor)
                
                # Store the sample
                self.data_samples.append((x_s_tensor, x_t_tensor, y_tensor))
                total_samples += 1
                
                if (i + 1) % 10 == 0:
                    print(f"    Generated {i + 1}/{n_samples_per_entry} samples for entry {entry_idx + 1}")
            
            # Log entry statistics
            if entry_y_tensors:
                entry_tensor = torch.stack(entry_y_tensors)
                y0_values = entry_tensor[:, 0]  # soft_success values
                y1_values = entry_tensor[:, 1]  # cost values
                
                print(f"  Entry {entry_idx + 1} statistics:")
                print(f"    Success rate: {y0_values.mean().item():.3f} (range: {y0_values.min().item():.3f}-{y0_values.max().item():.3f})")
                print(f"    Average cost: {y1_values.mean().item():.1f} (range: {y1_values.min().item():.1f}-{y1_values.max().item():.1f})")
        
        print(f"Generated {total_samples} total samples from {len(dataset_entries)} dataset entries")
        self._save_data(precision_level)
    
    def _load_checkout_config(self):
        """Load checkout configuration for the current problem."""
        checkout_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/checkouts/{self.cfg.dir}.yaml"
        if os.path.exists(checkout_path):
            with open(checkout_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Checkout config not found at {checkout_path}")
            return {}
    
    def _dict_to_tensor(self, param_dict):
        """Convert parameter dictionary to tensor using consistent parameter ordering."""
        
        # Define consistent parameter orders based on YAML config files
        PARAMETER_ORDERS = {
            # Heat 1D static parameters (from heat_1d/p1.yaml)
            'heat_1d_static': ['L', 'k', 'h', 'rho', 'cp', 'T_inf', 'T_init'],
            # Heat 1D tunable parameters
            'heat_1d_tunable': ['n_space', 'cfl'],
            
            # Heat steady 2D static parameters (from heat_steady_2d/p1.yaml) 
            #'heat_steady_2d_static': ['T_top', 'T_bottom', 'T_left', 'T_right', 'record_dt', 'end_frame', 'precision_level'],
            # Heat steady 2D tunable parameters
            #'heat_steady_2d_tunable': ['dx', 'relax', 'error_threshold', 'T_init'],
            
            # Burgers 1D static parameters (case only as one-hot)
            'burgers_1d_static': ['case'],
            # Burgers 1D tunable parameters  
            'burgers_1d_tunable': ['cfl', 'n_space', 'k', 'beta'],
            
            # Euler 1D static parameters (from euler_1d/p1.yaml)
            'euler_1d_static': ['L', 'gamma', 'case', 'precision_level'],
            # Euler 1D tunable parameters
            'euler_1d_tunable': ['cfl', 'beta', 'k', 'n_space'],
            
            # Common fallback order (sorted alphabetically)
            'common': []  # Will be filled dynamically with sorted keys
        }
        
        # Define possible values for categorical parameters
        CATEGORICAL_VALUES = {
            'case_euler': ['sod', 'lax', 'mach_3'],  # Euler 1D case types
            'case_burgers': ['sin', 'rarefaction', 'sod', 'double_shock', 'blast']  # Burgers 1D case types
        }
        
        # Determine which parameter order to use based on problem type and parameters present
        param_keys = set(param_dict.keys())
        
        # Try to match to specific problem types first
        if 'L' in param_keys and 'h' in param_keys:
            # Heat 1D static parameters
            key_order = PARAMETER_ORDERS['heat_1d_static']
        elif 'L' in param_keys and 'gamma' in param_keys:
            # Euler 1D static parameters
            key_order = PARAMETER_ORDERS['euler_1d_static']
        elif 'case' in param_keys and len(param_keys) == 1:
            # Burgers 1D static parameters - only case
            key_order = PARAMETER_ORDERS['burgers_1d_static']
        #elif 'T_top' in param_keys:
        #    # Heat steady 2D static parameters
        #    key_order = PARAMETER_ORDERS['heat_steady_2d_static']
        elif 'n_space' in param_keys and 'cfl' in param_keys and len(param_keys) == 2:
            # Heat 1D tunable parameters
            key_order = PARAMETER_ORDERS['heat_1d_tunable']
        elif 'cfl' in param_keys and 'beta' in param_keys and 'k' in param_keys and 'n_space' in param_keys:
            # Check if this is burgers_1d or euler_1d tunable parameters
            # Burgers 1D has 4 params: cfl, n_space, k, beta
            # Euler 1D has 4 params: cfl, beta, k, n_space
            if len(param_keys) == 4:
                # Use burgers_1d ordering for now - they're the same anyway
                key_order = PARAMETER_ORDERS['burgers_1d_tunable']
            else:
                key_order = PARAMETER_ORDERS['euler_1d_tunable']
        elif 'cfl' in param_keys and 'beta' in param_keys:
            # Euler 1D tunable parameters (partial)
            key_order = PARAMETER_ORDERS['euler_1d_tunable']
        #elif 'dx' in param_keys and 'relax' in param_keys:
        #    # Heat steady 2D tunable parameters
        #    key_order = PARAMETER_ORDERS['heat_steady_2d_tunable']
        else:
            # Use alphabetically sorted keys as fallback
            key_order = sorted(param_keys)
        
        # Extract values in the defined order
        values = []
        for key in key_order:
            if key in param_dict:
                val = param_dict[key]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif key == 'case':
                    # Handle categorical parameters with one-hot encoding
                    # For burgers_1d, always use the burgers case list for consistent tensor shape
                    if self.problem == "burgers_1d":
                        possible_values = CATEGORICAL_VALUES['case_burgers']
                    elif str(val) in CATEGORICAL_VALUES['case_euler']:
                        possible_values = CATEGORICAL_VALUES['case_euler']
                    elif str(val) in CATEGORICAL_VALUES['case_burgers']:
                        possible_values = CATEGORICAL_VALUES['case_burgers']
                    else:
                        raise ValueError(f"Unknown case value '{val}' for parameter '{key}'. Must be one of euler cases {CATEGORICAL_VALUES['case_euler']} or burgers cases {CATEGORICAL_VALUES['case_burgers']}")
                    
                    one_hot = [0.0] * len(possible_values)
                    if str(val) in possible_values:
                        one_hot[possible_values.index(str(val))] = 1.0
                    values.extend(one_hot)
                else:
                    # For other string types, raise an error to avoid silent issues
                    raise ValueError(f"Unknown categorical parameter '{key}' with value '{val}'. Please add it to CATEGORICAL_VALUES.")
        
        return torch.tensor(values, dtype=torch.float32)
    
    def _save_data(self, percision_level):
        """Save the generated data split into train (90%) and test (10%) sets."""
        if not self.data_samples:
            print("No data samples to save")
            return
            
        # Check if append mode is enabled
        append_mode = getattr(self.cfg, 'append_mode', False)
            
        # Create train and test output directories
        train_dir = os.path.join(self.cfg.output_dir, percision_level, "train")
        test_dir = os.path.join(self.cfg.output_dir, percision_level, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Create filename based on problem and tasks
        tasks_str = "_".join(self.cfg.tasks)
        filename = f"{self.problem}_{tasks_str}_samples.pt"
        
        train_filepath = os.path.join(train_dir, filename)
        test_filepath = os.path.join(test_dir, filename)
        
        # Handle append mode
        if append_mode and os.path.exists(train_filepath) and os.path.exists(test_filepath):
            print(f"Append mode enabled. Loading existing data from {train_filepath} and {test_filepath}")
            
            # Load existing data
            existing_train_samples = torch.load(train_filepath)
            existing_test_samples = torch.load(test_filepath)
            
            print(f"Found {len(existing_train_samples)} existing train samples and {len(existing_test_samples)} existing test samples")
            
            # Combine existing with new data
            all_existing_samples = existing_train_samples + existing_test_samples
            combined_samples = all_existing_samples + self.data_samples
            
            print(f"Combined {len(all_existing_samples)} existing samples with {len(self.data_samples)} new samples = {len(combined_samples)} total samples")
            
            # Shuffle the combined data for random split
            import random
            random.seed(42)  # For reproducibility
            shuffled_samples = combined_samples.copy()
            random.shuffle(shuffled_samples)
            
        else:
            # Normal mode - just use new data
            if append_mode:
                print("Append mode enabled but no existing files found. Creating new files.")
            
            # Shuffle the data for random split
            import random
            random.seed(42)  # For reproducibility
            shuffled_samples = self.data_samples.copy()
            random.shuffle(shuffled_samples)
        
        # Split data 90/10
        total_samples = len(shuffled_samples)
        train_size = int(0.9 * total_samples)
        
        train_samples = shuffled_samples[:train_size]
        test_samples = shuffled_samples[train_size:]
        
        print(f"Splitting {total_samples} samples: {len(train_samples)} train, {len(test_samples)} test")
        
        # Save train data
        torch.save(train_samples, train_filepath)
        print(f"Saved {len(train_samples)} training samples to {train_filepath}")
        
        # Save test data
        torch.save(test_samples, test_filepath)
        print(f"Saved {len(test_samples)} test samples to {test_filepath}")
        
        # Save metadata for both splits (only in non-append mode or when no existing files)
        if not append_mode or not (os.path.exists(train_filepath.replace('.pt', '_metadata.json')) and 
                                   os.path.exists(test_filepath.replace('.pt', '_metadata.json'))):
            
            base_metadata = {
                'problem': self.problem,
                'tasks': list(self.cfg.tasks),
                'sample_shape': {
                    'x_s_dim': self.data_samples[0][0].shape[0] if self.data_samples else 0,
                    'x_t_dim': self.data_samples[0][1].shape[0] if self.data_samples else 0,
                    'y_dim': self.data_samples[0][2].shape[0] if self.data_samples else 0,
                }
            }
            
            # Train metadata
            train_metadata = base_metadata.copy()
            train_metadata['n_samples'] = len(train_samples)
            train_metadata['split'] = 'train'
            
            train_metadata_file = train_filepath.replace('.pt', '_metadata.json')
            import json
            with open(train_metadata_file, 'w') as f:
                json.dump(train_metadata, f, indent=2)
            print(f"Saved train metadata to {train_metadata_file}")
            
            # Test metadata
            test_metadata = base_metadata.copy()
            test_metadata['n_samples'] = len(test_samples)
            test_metadata['split'] = 'test'
            
            test_metadata_file = test_filepath.replace('.pt', '_metadata.json')
            with open(test_metadata_file, 'w') as f:
                json.dump(test_metadata, f, indent=2)
            print(f"Saved test metadata to {test_metadata_file}")
        else:
            print("Append mode: Skipped metadata update (existing metadata files preserved)")
    
    @staticmethod
    def _generate_tunable(problem, task, sampling_method="random", fixed_params=None):
        """
        Generate tunable parameters for the given problem and task using specified sampling method.
        
        Args:
            problem (str): Problem type (e.g., "heat_1d", "heat_steady_2d", "burgers_1d")
            task (str): Task type (e.g., "cfl", "n_space", "dx", etc.)
            sampling_method (str): "random" or "linear"
            fixed_params (dict): Fixed non-target parameters
            
        Returns:
            dict: Dictionary containing sampled tunable parameters
        """
        if sampling_method == "random":
            return SimRewardGenerator._generate_random_tunable(problem, task, fixed_params)
        elif sampling_method == "linear":
            raise ValueError("Linear sampling should use _generate_linear_tunable_samples method")
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    @staticmethod
    def _generate_random_tunable(problem, task, fixed_params=None):
        """Generate random tunable parameters with constant values for non-tuned parameters."""
        import numpy as np
        
        if fixed_params is None:
            fixed_params = {}
            
        params = fixed_params.copy()
        
        if problem == "1D_heat_transfer":
            if task == "cfl":
                params['cfl'] = np.random.uniform(0.01, 1.0)
                params.setdefault('n_space', 100)
            elif task == "n_space":
                params.setdefault('cfl', 1.0)
                params['n_space'] = np.random.randint(20, 500)
        
        elif problem == "heat_steady_2d":
            if task == "dx":
                params['dx'] = np.random.uniform(0.001, 0.1)
                params.setdefault('relax', 1.0)
                params.setdefault('error_threshold', 1e-7)
                params.setdefault('T_init', 0.25)
            elif task == "error_threshold":
                params.setdefault('dx', 0.005)
                params.setdefault('relax', 1.0)
                params['error_threshold'] = 10**np.random.uniform(-10, -4)
                params.setdefault('T_init', 0.25)
            elif task == "relax":
                params.setdefault('dx', 0.005)
                params['relax'] = np.random.uniform(0.1, 1.9)
                params.setdefault('error_threshold', 1e-7)
                params.setdefault('T_init', 0.25)
            elif task == "t_init":
                params.setdefault('dx', 0.005)
                params.setdefault('relax', 1.0)
                params.setdefault('error_threshold', 1e-7)
                params['T_init'] = np.random.uniform(-1.0, 1.0)
        
        elif problem == "burgers_1d":
            if task == "cfl":
                params['cfl'] = np.random.uniform(0.01, 1.0)
                params.setdefault('n_space', 256)
                params.setdefault('k', -1.0)
                params.setdefault('beta', 1.0)
            elif task == "n_space":
                params.setdefault('cfl', 0.25)
                params['n_space'] = np.random.randint(64, 2049)
                params.setdefault('k', -1.0)
                params.setdefault('beta', 1.0)
            elif task == "k":
                params.setdefault('cfl', 1.0)
                params.setdefault('n_space', 256)
                params['k'] = np.random.uniform(-1.0, 1.0)
                params.setdefault('beta', 1.0)
            elif task == "beta":
                params.setdefault('cfl', 1.0)
                params.setdefault('n_space', 256)
                params.setdefault('k', -1.0)
                params['beta'] = np.random.uniform(1.0, 2.0)
                
        elif problem == "euler_1d":
            if task == "cfl":
                params['cfl'] = np.random.uniform(0.01, 1.0)
                params.setdefault('n_space', 256)
                params.setdefault('k', -1.0)
                params.setdefault('beta', 1.0)
            elif task == "n_space":
                params.setdefault('cfl', 1.0)
                params['n_space'] = np.random.randint(64, 2049)
                params.setdefault('k', -1.0)
                params.setdefault('beta', 1.0)
            elif task == "k":
                params.setdefault('cfl', 1.0)
                params.setdefault('n_space', 256)
                params['k'] = np.random.uniform(-1.0, 1.0)
                params.setdefault('beta', 1.0)
            elif task == "beta":
                params.setdefault('cfl', 1.0)
                params.setdefault('n_space', 256)
                params.setdefault('k', -1.0)
                params['beta'] = np.random.uniform(1.0, 2.0)
        
        else:
            raise ValueError(f"Unknown problem type: {problem}")
        
        # Round all float parameters to 6 digits for consistency
        for key, value in params.items():
            if isinstance(value, (float, np.floating)):
                params[key] = round(value, 6)
            
        return params
    
    @staticmethod
    def _generate_linear_tunable_samples(problem, task, n_samples, fixed_params=None):
        """
        Generate linear samples for tunable parameters within their parameter bounds.
        
        Args:
            problem (str): Problem type (e.g., "heat_1d", "heat_steady_2d", "burgers_1d")
            task (str): Task type (e.g., "cfl", "n_space", "dx", etc.)
            n_samples (int): Number of linear samples to generate
            fixed_params (dict): Fixed non-target parameters
            
        Returns:
            list: List of dictionaries containing linearly sampled tunable parameters
        """
        import numpy as np
        
        if fixed_params is None:
            fixed_params = {}
        
        samples = []
        
        # Define parameter bounds for each problem and task
        param_bounds = SimRewardGenerator._get_parameter_bounds(problem, task)
        
        # Get the target parameter being tuned
        target_param = SimRewardGenerator._get_target_parameter(problem, task)
        
        if target_param not in param_bounds:
            raise ValueError(f"Target parameter {target_param} not found in bounds for {problem}/{task}")
        
        # Generate linear samples for the target parameter
        param_min, param_max = param_bounds[target_param]
        if target_param in ['n_space'] and problem in ['1D_heat_transfer', 'burgers_1d', 'euler_1d']:
            # Integer parameter - use linspace and convert to int
            linear_values = np.linspace(param_min, param_max, n_samples, dtype=int)
        else:
            # Float parameter
            linear_values = np.linspace(param_min, param_max, n_samples)
        
        for value in linear_values:
            params = fixed_params.copy()
            
            # Set the target parameter to the linear value
            if target_param in ['n_space'] and problem in ['1D_heat_transfer', 'burgers_1d', 'euler_1d']:
                params[target_param] = int(value)
            else:
                params[target_param] = round(float(value), 4)
            
            samples.append(params)
        
        return samples
    
    @staticmethod
    def _generate_random_tunable_samples(problem, task, n_samples, fixed_params=None):
        """
        Generate random samples for tunable parameters within their parameter bounds.
        
        Args:
            problem (str): Problem type (e.g., "1D_heat_transfer", "heat_steady_2d", "burgers_1d")
            task (str): Task type (e.g., "cfl", "n_space", "dx", etc.)
            n_samples (int): Number of random samples to generate
            fixed_params (dict): Fixed non-target parameters
            
        Returns:
            list: List of dictionaries containing randomly sampled tunable parameters
        """
        import numpy as np
        
        if fixed_params is None:
            fixed_params = {}
        
        samples = []
        
        # Define parameter bounds for each problem and task
        param_bounds = SimRewardGenerator._get_parameter_bounds(problem, task)
        
        # Get the target parameter being tuned
        target_param = SimRewardGenerator._get_target_parameter(problem, task)
        
        if target_param not in param_bounds:
            raise ValueError(f"Target parameter {target_param} not found in bounds for {problem}/{task}")
        
        # Generate random samples for the target parameter
        param_min, param_max = param_bounds[target_param]
        
        for _ in range(n_samples):
            params = fixed_params.copy()
            
            # Sample the target parameter randomly within bounds
            if target_param in ['n_space'] and problem in ['1D_heat_transfer', 'burgers_1d', 'euler_1d']:
                # Integer parameter - use randint
                params[target_param] = np.random.randint(param_min, param_max + 1)
            elif target_param == 'error_threshold':
                # Log-uniform sampling for error_threshold
                log_min = np.log10(param_min)
                log_max = np.log10(param_max)
                log_value = np.random.uniform(log_min, log_max)
                params[target_param] = round(10**log_value, 10)
            else:
                # Float parameter - use uniform sampling
                params[target_param] = round(np.random.uniform(param_min, param_max), 4)
            
            samples.append(params)
        
        return samples
    
    @staticmethod
    def _get_parameter_bounds(problem, task):
        """Get parameter bounds for linear sampling."""
        bounds = {}
        
        if problem == "1D_heat_transfer":
            if task == "cfl":
                bounds['cfl'] = (0.01, 1.0)
            elif task == "n_space":
                bounds['n_space'] = (64, 1024)
                
        elif problem == "heat_steady_2d":
            if task == "dx":
                bounds['dx'] = (0.001, 0.1)
            elif task == "error_threshold":
                bounds['error_threshold'] = (1e-10, 1e-4)
            elif task == "relax":
                bounds['relax'] = (0.1, 1.9)
            elif task == "t_init":
                bounds['T_init'] = (-1.0, 1.0)
                
        elif problem == "burgers_1d":
            if task == "cfl":
                bounds['cfl'] = (0.01, 1.0)
            elif task == "n_space":
                bounds['n_space'] = (64, 2048)
            elif task == "k":
                bounds['k'] = (-1.0, 1.0)
            elif task == "beta":
                bounds['beta'] = (1.0, 2.0)
                
        elif problem == "euler_1d":
            if task == "cfl":
                bounds['cfl'] = (0.01, 1.0)
            elif task == "n_space":
                bounds['n_space'] = (64, 2048)
            elif task == "k":
                bounds['k'] = (-1.0, 1.0)
            elif task == "beta":
                bounds['beta'] = (1.0, 2.0)
        
        return bounds
    
    @staticmethod
    def _get_target_parameter(problem, task):
        """Get the target parameter name being tuned for a given problem/task."""
        if problem == "1D_heat_transfer":
            return task  # task is the parameter name (cfl, n_space)
        elif problem == "heat_steady_2d":
            if task == "t_init":
                return "T_init"
            else:
                return task  # dx, error_threshold, relax
        elif problem == "burgers_1d":
            return task  # cfl, n_space, k, beta
        elif problem == "euler_1d":
            return task  # cfl, n_space, k, beta
        else:
            raise ValueError(f"Unknown problem: {problem}")
    


@hydra.main(version_base=None, config_path="/home/ubuntu/dev/src/template_trainer/dataset/gen_configs/", config_name="1D_heat_transfer.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main function for data generation using Hydra configuration.
    
    Expected config structure:
    ```yaml
    problem: "1D_heat_transfer"  # or "2D_heat_transfer", "1D_burgers"
    tasks:
      - "cfl"
      - "n_space"
    n_sample: 1000
    profile_path: "/path/to/profiles"
    gt_dir: "/path/to/ground/truth"
    output_dir: "/path/to/output"
    ```
    """
    print("Starting data generation with config:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize the generator
    generator = SimRewardGenerator(cfg, cfg.problem)
    
    # Generate data with specified precision level
    precision_level = getattr(cfg, 'precision_level', ['medium'])  # Default to 'medium' if not specified
    modes = getattr(cfg, 'sampling_method', ['random'])
    '''for mode in modes:
        generator.generate_data_from_dataset("euler_1d", "cfl", precision_level, mode="zero_shot", sampling_method=mode)'''
    existing_path = getattr(cfg, 'from_existing', None)
    if existing_path is None:
        modes = getattr(cfg, 'sampling_method', ['random'])
        for mode in modes:
            generator.generate_data(precision_level, mode)
    else:
        train_path = f"{existing_path}/train/euler_1d_cfl_samples.pt"
        #output_train = train_path.replace("high", precision_level)
        test_path = f"{existing_path}/test/euler_1d_cfl_samples.pt"
        #output_test = test_path.replace("high", precision_level)

        generator.generate_from_existing(train_path, precision_level, "high")
        generator.generate_from_existing(test_path, precision_level, "high")
        
    print("Data generation completed!")


if __name__ == "__main__":
    main()