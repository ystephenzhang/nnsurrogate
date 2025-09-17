from typing import Literal
import re
import json
import ast
import random
random.seed(42)

import numpy as np
import sys
import os
import torch
import glob
import yaml
from types import SimpleNamespace
import pdb

sys.path.append('/home/ubuntu/dev/src')
from general_utils import INFO, extract_static, soft_success, reverse_preproc, get_tolerance

# Load task descriptions from the extracted JSON file
#with open('/home/ubuntu/dev/src/opro/optimization/task_descriptions.json', 'r') as f:
BASE_PROMPTS = ""

# Load physics parameter templates
#with open('/home/ubuntu/dev/src/opro/optimization/physics_templates.json', 'r') as f:
PHYSICS_TEMPLATES = ""

from template_trainer.model.sim import SimRewardModel

# Global model cache to avoid reloading
_MODEL_CACHE = {}

# Map problem names to directory names
dir_mapping = {
    "1D_heat_transfer": "heat_1d",
    "2D_heat_transfer": "heat_steady_2d", 
    "1D_burgers": "burgers_1d",
    "euler_1d": "euler_1d",
    "ns_channel_2d": "ns_channel_2d",
    "ns_transient_2d": "ns_transient_2d"
}
def get_description(problem, task, profile):
    messages_path = f"/home/ubuntu/dev/SimulCost-Bench/data/{problem}/human_write/{task}_zero_shot_dataset.json"
    
    pid = int(profile[1:])
    with open(messages_path, 'r') as f:
        data = json.load(f)
        
    system = data[pid]["messages"][0]["content"]
    ins = data[pid]["messages"][1]["content"]

    return system + "\n\n" + ins

def _get_description(problem, task, profile):
    """
    Get the complete task description including physics parameters from profile.
    
    Args:
        problem: Problem type (e.g., "1D_heat_transfer", "2D_heat_transfer", "1D_burgers")
        task: Task parameter (e.g., "cfl", "n_space", "dx", etc.)
        profile: Profile identifier (e.g., "1", "2", etc.)
        
    Returns:
        str: Complete task description with physics parameters filled in
    """
    if problem not in BASE_PROMPTS:
        raise ValueError(f"Unknown problem: {problem}")
    if task not in BASE_PROMPTS[problem]:
        raise ValueError(f"Unknown task {task} for problem {problem}")
    
    # Get base task description
    base_description = BASE_PROMPTS[problem][task]
    
    # Get physics template and fill it with profile data
    physics_section = fill_physics_template(problem, task, profile)
    
    # Combine base description with physics parameters
    full_description = base_description + "\n\n" + physics_section
    
    return full_description

def _fill_physics_template(problem, task, profile):
    """
    Fill physics parameter template with values from profile.yaml.
    
    Args:
        problem: Problem type
        task: Task parameter
        profile: Profile identifier
        
    Returns:
        str: Physics section with actual parameter values
    """
    if problem not in PHYSICS_TEMPLATES:
        raise ValueError(f"Unknown problem: {problem}")
    if task not in PHYSICS_TEMPLATES[problem]:
        raise ValueError(f"Unknown task {task} for problem {problem}")
    
    # Get the template
    template = PHYSICS_TEMPLATES[problem][task]
    
    # Load profile data
    profile_data = load_profile_data(problem, profile)
    
    # Fill template with actual values
    filled_template = template
    
    if problem == "1D_heat_transfer":
        # Template: "Problem: 1D transient heat conduction in a wall with:\n- Wall thickness: 0.150000 m\n..."
        # Fill with actual values from profile
        filled_template = f"""Problem: 1D transient heat conduction in a wall with:
    - Wall thickness: {profile_data.get('L', 0.15):.6f} m
    - Left boundary: Convection (h = {profile_data.get('h', 1000.0):.3f} W/m^2-K, T_inf = {profile_data.get('T_inf', 12.0):.2f} C)
    - Right boundary: Insulated (zero heat flux)
    - Initial temperature: {profile_data.get('T_init', 25.0):.2f} C (uniform)
    - Thermal conductivity: {profile_data.get('k', 1.0):.4f} W/m-K
    - Specific heat: {profile_data.get('cp', 1000.0):.1f} J/kg-K
    - Density: {profile_data.get('rho', 1500.0):.1f} kg/m^3
    - Recording interval: {profile_data.get('record_dt', 10.0):.4f} s"""
            
    elif problem == "2D_heat_transfer":
        # Template: "Physical Parameters:\n- T_top: 1.0\n..."
        # Fill with actual values from profile
        filled_template = f"""Physical Parameters:
    - T_top: {profile_data.get('T_top', 1.0)}
    - T_bottom: {profile_data.get('T_bottom', 0)}
    - T_left: {profile_data.get('T_left', 0)}
    - T_right: {profile_data.get('T_right', 0)}"""
        
    elif problem == "1D_burgers":
        # Template includes convergence check info, so we keep that and just fill physics params
        filled_template = f"""Physical Parameters:
- Domain length: {profile_data.get('L', 2.0)}
- Case: {profile_data.get('case', 'sin')}

Convergence Check:
- The current CFL value will be halved (veri_cfl = cfl / 2), and the simulation is rerun with veri_cfl.
- Errors between the two simulations are computed to assess convergence.
- Convergence is confirmed if the following validation criteria are satisfied."""
    
    return filled_template

def load_profile_data(problem, profile):
    """
    Load parameter values from profile.yaml file.
    
    Args:
        problem: Problem type
        profile: Profile identifier
        
    Returns:
        dict: Parameter values from the profile
    """
    
    
    if problem not in dir_mapping:
        raise ValueError(f"Unknown problem: {problem}")
    
    dir_name = dir_mapping[problem]
    # Profile files are named p1.yaml, p2.yaml, etc.
    profile_file = f"p{profile}.yaml"
    profile_path = f"/home/ubuntu/codebase/SimulCost-Bench/costsci_tools/run_configs/{dir_name}/{profile_file}"
    
    try:
        with open(profile_path, 'r') as f:
            profile_data = yaml.safe_load(f)
        return profile_data
    except FileNotFoundError:
        print(f"Profile file not found: {profile_path}")
        return {}
    except Exception as e:
        print(f"Error loading profile {profile_path}: {e}")
        return {}

def load_surrogate_model(problem, tolerance, config=None):
    """
    Load trained surrogate model for the given problem/task/profile.
    
    Args:
        problem: Problem type (e.g., "1D_heat_transfer")
        task: Task parameter (e.g., "cfl") 
        profile: Profile identifier
        config: Optional config object containing model_path and step information
        
    Returns:
        torch.nn.Module: Loaded surrogate model
    """
    cache_key = f"{problem}_{tolerance}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
        params_path = config_dict["best_checkpoint_path"]
    '''try:
        restore_dir = config_dict["direct_restore_dir"]
        params_path = f"{restore_dir}/best_params.pth"
    except:"'''


    
    '''else:
        # Fallback to finding the model directory (original approach)
        model_base_dir = f"/home/ubuntu/dev/src/template_trainer/output/Surrogate-{problem}"
        if not os.path.exists(model_base_dir):
            raise FileNotFoundError(f"Surrogate model directory not found: {model_base_dir}")
        
        # Find the most recent run directory
        run_dirs = glob.glob(os.path.join(model_base_dir, "*/"))
        if not run_dirs:
            raise FileNotFoundError(f"No training runs found in {model_base_dir}")
        
        # Get the most recent run (assuming timestamp-based naming)
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        
        # Find the model checkpoint with highest step count
        param_files = glob.glob(os.path.join(latest_run_dir, "*_params.pth"))
        if not param_files:
            raise FileNotFoundError(f"No model checkpoints found in {latest_run_dir}")
        
        # Extract step numbers and find the highest one
        step_numbers = []
        for param_file in param_files:
            basename = os.path.basename(param_file)
            step_str = basename.split('_')[0]
            try:
                step_numbers.append((int(step_str), param_file))
            except ValueError:
                continue
        
        if not step_numbers:
            raise ValueError(f"No valid step numbers found in checkpoint files")
        
        # Use the highest step checkpoint
        _, params_path = max(step_numbers)'''
    
    # Convert config dict to namespace for model initialization
    model_config = SimpleNamespace(**config_dict['model'])
    
    # Create and load the model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = SimRewardModel(model_config).to(device)
    
    # Load trained parameters using restore() style approach
    model.load_state_dict(torch.load(params_path, map_location=device))
    model.eval()
    
    # Cache the loaded model
    _MODEL_CACHE[cache_key] = model
    
    print(f"Loaded surrogate model from {params_path}")
    return model

def prepare_model_input(params, problem, profile, verifier=None):
    """
    Prepare input tensors for the surrogate model.
    
    Args:
        params: Dictionary of simulation parameters
        problem: Problem type
        profile: Profile identifier
        verifier: Verifier object (optional, needed for euler_1d tolerance)
        
    Returns:
        tuple: (static_tensor, tunable_tensor) ready for model input
    """
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    dir_name = dir_mapping[problem]
    # Get static parameters from the profile configuration
    profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{dir_name}/{profile}.yaml"
    static_params = extract_static(profile_path)
    
    # Define parameter order based on problem type
    if problem == "1D_heat_transfer":
        # Static parameters: [L, k, h, rho, cp, T_inf, T_init, level_encoding] (matches heat_1d_static_mapper)
        static_order = ['L', 'k', 'h', 'rho', 'cp', 'T_inf', 'T_init']
        # Add level encoding like in training data
        level_map = {'low': 0, 'medium': 1, 'high': 2}
        level_encoding = level_map.get(verifier.tolerance if verifier and hasattr(verifier, 'tolerance') else 'medium', 1)
        
        # Tunable parameters: [cfl, n_space] (matches heat_1d_tunable_mapper)
        tunable_order = ['cfl', 'n_space']
    elif problem == "2D_heat_transfer":
        # Static parameters: [T_top, T_bottom, T_left, T_right]
        static_order = ['T_top', 'T_bottom', 'T_left', 'T_right']
        # Tunable parameters: [dx, relax, error_threshold, t_init]
        tunable_order = ['dx', 'relax', 'error_threshold', 't_init']
    elif problem == "1D_burgers":
        # Static parameters: [L, case]
        static_order = ['L', 'case']
        # Tunable parameters: [cfl, k, w]
        tunable_order = ['cfl', 'k', 'w']
    elif problem == "euler_1d":
        # Import get_tolerance from general_utils  
        from general_utils import get_tolerance
        
        # Static parameters: case_one_hot (3 elements) + [level_encoding] (matches euler_1d_static_mapper)
        # Note: case mapping must match exactly: {"sod": 0, "lax": 1, "mach_3": 2}
        case_mapping = {
            "sod": 0,
            "lax": 1, 
            "mach_3": 2
        }
        
        case_str = str(static_params.get('case', 'sod'))
        case_one_hot = [0.0, 0.0, 0.0]
        if case_str in case_mapping:
            case_one_hot[case_mapping[case_str]] = 1.0
        
        # Use level encoding instead of tolerance value
        level_map = {'low': 0, 'medium': 1, 'high': 2}
        level_encoding = level_map.get(verifier.tolerance if verifier and hasattr(verifier, 'tolerance') else 'medium', 1)
        
        # Build static_values: case_one_hot + [level_encoding] 
        static_values = case_one_hot + [level_encoding]
        
        # Tunable parameters: [cfl, n_space, beta, k] (matches euler_1d_tunable_mapper order)
        tunable_order = ['cfl', 'n_space', 'beta', 'k']
        
    elif problem == "ns_channel_2d":
        # Static parameters: [length, breadth, mu, rho, level_encoding] + boundary_onehot (matches ns_channel_2d_static_mapper)
        
        # One-hot encoding for boundary condition
        boundary_condition = static_params.get('boundary_condition', 'channel_flow')
        boundary_conditions = ['channel_flow', 'back_stair_flow', 'expansion_channel', 'cube_driven_flow']
        boundary_onehot = [0.0] * len(boundary_conditions)
        if boundary_condition in boundary_conditions:
            boundary_onehot[boundary_conditions.index(boundary_condition)] = 1.0
        
        # Use level encoding
        level_map = {'low': 0, 'medium': 1, 'high': 2}
        level_encoding = level_map.get(verifier.tolerance if verifier and hasattr(verifier, 'tolerance') else 'medium', 1)
        
        # Static fluid properties and parameters
        static_base = [
            static_params.get('length', 20.0),
            static_params.get('breadth', 1.0),
            static_params.get('mu', 0.01),
            static_params.get('rho', 1.0),
            level_encoding
        ]
        
        # Combine static_base + boundary_onehot
        static_values = static_base + boundary_onehot
        
        # Tunable parameters: [mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold]
        # (matches ns_channel_2d_tunable_mapper order)
        tunable_order = ['mesh_x', 'mesh_y', 'omega_u', 'omega_v', 'omega_p', 'diff_u_threshold', 'diff_v_threshold', 'res_iter_v_threshold']
        
    elif problem == "ns_transient_2d":
        # Static parameters: boundary_onehot (6 elements) + [reynolds_encoding, level_encoding] (matches ns_transient_2d_static_mapper)
        
        # One-hot encoding for boundary condition (1-6)
        boundary_condition = static_params.get('boundary_condition', 1)
        boundary_onehot = [0.0] * 6
        if 1 <= boundary_condition <= 6:
            boundary_onehot[boundary_condition - 1] = 1.0  # Convert to 0-indexed
        
        # Reynolds number encoding (1k=0, 100k=1)
        reynolds_num = static_params.get('reynolds_num', 1000.0)
        reynolds_encoding = 1 if reynolds_num >= 50000 else 0  # 100k vs 1k
        
        # Use level encoding
        level_map = {'low': 0, 'medium': 1, 'high': 2}
        level_encoding = level_map.get(verifier.tolerance if verifier and hasattr(verifier, 'tolerance') else 'medium', 1)
        
        # Combine boundary_onehot + [reynolds_encoding, level_encoding]
        static_values = boundary_onehot + [reynolds_encoding, level_encoding]
        
        # Tunable parameters: [cfl, resolution, relaxation_factor, residual_threshold] (matches ns_transient_2d_tunable_mapper order)
        tunable_order = ['cfl', 'resolution', 'relaxation_factor', 'residual_threshold']
        
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    
    # Create static feature tensor
    if problem == "1D_heat_transfer":
        # Include level encoding as the last element
        static_values = [static_params.get(key, 0.0) for key in static_order] + [level_encoding]
    elif problem == "euler_1d":
        pass  # static_values already set above
    elif problem == "ns_channel_2d":
        pass  # static_values already set above
    elif problem == "ns_transient_2d":
        pass  # static_values already set above
    else:
        static_values = [static_params.get(key, 0.0) for key in static_order]
    static_tensor = torch.tensor(static_values, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Create tunable feature tensor
    if problem == "ns_channel_2d":
        # Special handling for ns_channel_2d parameters
        tunable_values = []
        for key in tunable_order:
            if key == 'mesh_x' or key == 'mesh_y':
                # Convert to int then float for mesh parameters
                tunable_values.append(float(int(params.get(key, 64 if key == 'mesh_x' else 16))))
            elif key == 'res_iter_v_threshold':
                # Handle res_iter_v_threshold which can be string or float
                res_iter_v_threshold = params.get('res_iter_v_threshold', 'exp_decay')
                if isinstance(res_iter_v_threshold, str) and res_iter_v_threshold == 'exp_decay':
                    tunable_values.append(0.0)  # Use 0.0 to indicate exp_decay
                else:
                    tunable_values.append(float(res_iter_v_threshold))
            else:
                tunable_values.append(params.get(key, 0.0))
    elif problem == "ns_transient_2d":
        # Special handling for ns_transient_2d parameters
        tunable_values = []
        for key in tunable_order:
            if key == 'resolution':
                # Convert to int then float for resolution parameter
                tunable_values.append(float(int(params.get(key, 200))))
            else:
                # Default values: cfl=0.05, relaxation_factor=1.3, residual_threshold=0.01
                default_vals = {'cfl': 0.05, 'relaxation_factor': 1.3, 'residual_threshold': 0.01}
                tunable_values.append(params.get(key, default_vals.get(key, 0.0)))
    else:
        tunable_values = [params.get(key, 0.0) for key in tunable_order]
    
    tunable_tensor = torch.tensor(tunable_values, dtype=torch.float32, device=device).unsqueeze(0)
    
    return static_tensor, tunable_tensor

def evaluate(params,
             verifier,
             profile,
             qid,
             backend: Literal["surrogate", "ground_truth"]="surrogate",
             surrogate_config=None,
             use_soft_success=True,
             prev_cost=0,
             threshold=None):
    """
    Evaluate simulation parameters using the verifier.
    
    Args:
        params: Dictionary of parameters to evaluate
        verifier: Verifier object with metric method
        backend: Evaluation backend ("surrogate" or "ground_truth")
    
    Returns:
        float: Evaluation score (higher is better)
    """
    #try:
    #gt_success, gt_cost, gt_score = verifier.metric(params, profile, qid, soft_success=soft_success, prev_cost=prev_cost)
    if backend == "ground_truth":
        #pdb.set_trace()
        success, cost, score = verifier.metric(params, profile, qid, soft_success=use_soft_success, prev_cost=prev_cost)
        success = float(success)
        efficiency = verifier.best_costs[qid] / (1e-3 + cost)
            
    elif backend == "surrogate":
        # Use surrogate model for evaluation
        if surrogate_config == "random":
            success = random.uniform(0, 1)

            efficiency = random.uniform(0.1, 10)

            return success, efficiency, success * efficiency
        
        # Load the surrogate model
        model = load_surrogate_model(verifier.problem, verifier.tolerance, surrogate_config)
        # Prepare input tensors
        static_tensor, tunable_tensor = prepare_model_input(params, verifier.problem, profile, verifier)
        
        # Get model prediction
        with torch.no_grad():
            prediction = model((static_tensor, tunable_tensor))
            
            # prediction is a tensor of shape [batch_size, 2] containing [success, cost]
            # Extract success and cost predictions
            with open(surrogate_config, 'r') as f:
                cfg = yaml.safe_load(f)
            success_target = cfg["dataset"].get("success_target", "score")
            preproc_cost = cfg["dataset"].get("preproc_cost", "score")

            if success_target == "score":
                success = float(prediction[0, 0])  # First output is success probability 
            else:
                success = soft_success(float(prediction[0, 0]), get_tolerance(verifier.problem, verifier.tolerance))
            
            if preproc_cost:
                cost = reverse_preproc(float(prediction[0, 1]), params)
            else:
                cost= float(prediction[0, 1])     # Second output is cost
            
            #pdb.set_trace()
            if cost <= 0:
                if "n_space" in params:
                    n_space = int(params.get("n_space"))
                    cfl = params.get("cfl")
                    cost = 1.05 * n_space * n_space / cfl
                else:
                    cost = verifier.best_costs[qid]
            
            # Ensur"e reasonable bounds
            #success= max(0.0, min(1.0, success))  # Clamp between 0 and 1
            efficiency = verifier.best_costs[qid] / (1e-3 + cost + prev_cost)
            
            if not use_soft_success:
                success = 1.0 if success >= 0.99 else 0
            else:
                # Score combines success probability and cost efficiency  
                #success = 1.0 if success >= 0.99 else success 
                if threshold:
                    success = 1.0 if success >= threshold else success 
                else:
                    success = 1.0 if success >= 0.99 else success 
            
            score = success * efficiency
            
    else:
        raise ValueError(f"Unknown backend: {backend}")
        
    
    #return max(0, score)  # Ensure non-negative score
    return max(0, success), efficiency, max(0, score)
    #except Exception as e:
    #    print(f"Evaluation error: {e}")
    #    return 0.0
import copy
def sample_sols_refined(
    problem,
    task,
    verifier,
    profile,
    qid,
    level,
    num_samples=5,
    gt_sol=None,
    sampler="linear",
    backend="surrogate",
    surrogate_config=None,
    soft_succ=True,
    threshold=None,
    ratio=None
):
    if task not in INFO:
        raise ValueError(f"Unknown task: {task}")
    
    #task_info = INFO[task]
    #param_range = task_info["range"]
    task_info = {
        "n_space":{
            "range":{
                "low": [64, 128],
                "medium": [64, 512],
                "high": [64, 512]
            }
        },
        "mesh_x":{
            "range":{
                "low": [16, 64],
                "medium": [16, 128],
                "high": [16, 128]
            }
        },
        "resolution":{
            "range":{
                "low": [32, 100],
                "medium": [32, 100],
                "high": [64, 150]
            }
        }
            
    }
    param_info = task_info[task]
    param_type = param_info.get("type", "int")
    param_range = param_info["range"][level]
    
    solutions = []

    if sampler == "linear":
        # Linear sampling: evenly distribute samples across the parameter range
        for i in range(num_samples):
            if num_samples == 1:
                # If only one sample, use the middle of the range
                value = (param_range[0] + param_range[1]) / 2
            else:
                # Linearly interpolate between min and max
                alpha = i / (num_samples - 1)  # 0 to 1
                value = param_range[0] + alpha * (param_range[1] - param_range[0])
            
            if param_type == "int":
                value = int(round(value))
            else:  # float
                value = round(value, 4)
            
            params = copy.deepcopy(gt_sol)
            params[task] = value

            if problem == "ns_channel_2d":
                params["mesh_y"] = int(value * ratio)
            elif problem == "ns_transient_2d":
                params["resolution"] = 2 * (params["resolution"] // 2)

            if not threshold:
                success, efficiency, score = evaluate(params, verifier, profile, qid, backend=backend, surrogate_config=surrogate_config, use_soft_success=soft_succ)
            else:
                success, efficiency, score = evaluate(params, verifier, profile, qid, backend=backend, surrogate_config=surrogate_config, use_soft_success=soft_succ, threshold=threshold)
            solutions.append((params, success, efficiency))
    
    # Finally add dummy solution
    #dummy_success, dummy_efficiency, _ = evaluate(verifier.best_params[qid], verifier, profile, qid, backend=backend, surrogate_config=surrogate_config, soft_success=soft_succ)
    #solutions.append((verifier.best_params[qid], dummy_success, dummy_efficiency))

    return solutions

        
def sample_sols(problem, task, verifier, profile, num_samples=5, gt_sol=None, sampler="random", backend="ground_truth", surrogate_config=None, soft_succ=True):
    """
    Generate initial solution samples for optimization.
    
    Args:
        profile: Profile identifier
        problem: Problem type (e.g., "1D_heat_transfer", "2D_heat_transfer", "1D_burgers") 
        task: Task parameter to optimize (e.g., "cfl", "n_space", "dx", etc.)
        sampler: Sampling method ("random")
    
    Returns:
        List of (params_dict, score) tuples for initial solutions
    """
    if task not in INFO:
        raise ValueError(f"Unknown task: {task}")
    
    task_info = INFO[task]
    param_range = task_info["range"]
    param_type = task_info.get("type", "float")
    
    solutions = []
    if sampler == "random":
        for _ in range(num_samples):
            if param_type == "int":
                value = random.randint(int(param_range[0]), int(param_range[1]))
            else:  # float
                value = random.uniform(param_range[0], param_range[1])
            
            params = {task: round(value, 4)}
            
            # Add other required parameters with default values
            if problem == "1D_heat_transfer":
                if task != "cfl":
                    params["cfl"] = INFO["cfl"]["initial"]
                if task != "n_space":
                    params["n_space"] = INFO["n_space"]["initial"]
            elif problem == "2D_heat_transfer":
                for param in ["dx", "relax", "error_threshold", "t_init"]:
                    if param != task and param in INFO:
                        params[param] = INFO[param].get("initial", (INFO[param]["range"][0] + INFO[param]["range"][1]) / 2)
            elif problem == "1D_burgers":
                for param in ["cfl", "k", "w"]:
                    if param != task and param in INFO:
                        params[param] = INFO[param].get("initial", (INFO[param]["range"][0] + INFO[param]["range"][1]) / 2)
            
            success, cost, score = evaluate(params, verifier, profile, backend=backend, surrogate_config=surrogate_config, use_soft_success=soft_succ)
            solutions.append((params, score))
    
    elif sampler == "linear":
        # Linear sampling: evenly distribute samples across the parameter range
        for i in range(num_samples):
            if num_samples == 1:
                # If only one sample, use the middle of the range
                value = (param_range[0] + param_range[1]) / 2
            else:
                # Linearly interpolate between min and max
                alpha = i / (num_samples - 1)  # 0 to 1
                value = param_range[0] + alpha * (param_range[1] - param_range[0])
            
            if param_type == "int":
                value = int(round(value))
            else:  # float
                value = round(value, 4)
            
            params = copy.deepcopy(gt_sol)
            params[task] = value
            '''params = {task: value}
            
            # Add other required parameters with default values
            if problem == "1D_heat_transfer":
                if task != "cfl":
                    params["cfl"] = INFO["cfl"]["initial"]
                if task != "n_space":
                    params["n_space"] = INFO["n_space"]["initial"]
            elif problem == "2D_heat_transfer":
                for param in ["dx", "relax", "error_threshold", "t_init"]:
                    if param != task and param in INFO:
                        params[param] = INFO[param].get("initial", (INFO[param]["range"][0] + INFO[param]["range"][1]) / 2)
            elif problem == "1D_burgers":
                for param in ["cfl", "k", "w"]:
                    if param != task and param in INFO:
                        params[param] = INFO[param].get("initial", (INFO[param]["range"][0] + INFO[param]["range"][1]) / 2)'''
            
            success, cost, score = evaluate(params, verifier, profile, backend=backend, surrogate_config=surrogate_config, use_soft_success=soft_succ)
            solutions.append((params, score))
    
    elif sampler == "none":
        return solutions
    
    return solutions

def gen_meta_prompt(
    old_value_pairs_set,
    profile,
    problem,
    task,
    complete_description=None,
    max_num_pairs=100,
    target="refined",
    mode="zero-shot"
):
    """
    Generate meta-prompt for LLM optimization.
    
    Args:
        old_value_pairs_set: Set of (params, score) pairs from previous iterations
        profile: Profile identifier
        problem: Problem type 
        task: Task parameter to optimize
        max_num_pairs: Maximum number of examples to include in prompt
        
    Returns:
        str: Formatted meta-prompt for the LLM
    """
    if task not in INFO:
        raise ValueError(f"Unknown task: {task}")
    
    task_info = INFO[task]
    param_range = task_info["range"]
    
    # Get the complete task description with physics parameters from SimulCost-Bench
    if not complete_description:
        complete_description = get_description(problem, task, profile)
    meta_prompt = complete_description + "\n\n"
    
    # Convert set to list and sort by score (ascending for better performance - higher score is better)
    old_value_pairs = list(old_value_pairs_set)
   
    
    # Build the examples substring
    old_value_pairs_substr = ""

    if target == "score":
        old_value_pairs = sorted(old_value_pairs, key=lambda x: float(x[1]) * float(x[2]))[-max_num_pairs:]
        for params_str, success, cost in old_value_pairs:
            score = float(success) * float(cost)
            try:
                params = ast.literal_eval(params_str)
                param_value = params[task]
                old_value_pairs_substr += f"\n<{task}> {param_value} </{task}>\nperformance score:\n{score:.4f}\n"
            except:
                continue
        
        if old_value_pairs_substr != "":
            meta_prompt += f"Below are some previous {task} values and their performance scores. The values are arranged in ascending order based on their performance scores, where higher values are better."
        
            meta_prompt += "\n\n"
            meta_prompt += old_value_pairs_substr.strip()
            meta_prompt += "\n\n"
            meta_prompt += f"""\n\nOutput final answer in the requested format with a new {task} value that is different from all values above, and has a performance score higher than any of the above. Do note that the performance scores above are indicators of successfulness; you need to maximize it.
            """.strip()
              
    else:
        old_value_pairs = sorted(old_value_pairs, key=lambda x: (float(x[1]), float(x[2])))[-max_num_pairs:]
        for params_str, success, efficiency in old_value_pairs:
            try:
                params = ast.literal_eval(params_str)
                param_value = params[task]
                old_value_pairs_substr += f"\n<{task}> {param_value} </{task}>\n Accuracy Indicator:\n{float(success):.4f}\n Efficiency Indicator:\n{float(efficiency):.4f}\n"
            except:
                continue
        
        if old_value_pairs_substr != "":
            meta_prompt += f"Below are some previous {task} values and their simulation accuracy and efficiency indicators. The values are arranged in ascending order based on accuracy, where higher values indicate a closer simulation result to ground truth. The efficiency indicator is also important, where higher values mean a more cost-efficient {task} choice."
        
            meta_prompt += "\n\n"
            meta_prompt += old_value_pairs_substr.strip()
            meta_prompt += "\n\n"
            if mode == "zero-shot":
                meta_prompt += f"""\n\nOutput final answer in the requested format with a new {task} value that is different from all values above. You should first ensure an accurate simulation by achieving 1.0 in accuracy indicator, then gradually increase efficiency by choosing a coarser {task} value.
                """.strip()
            else:
                meta_prompt += f"""\n\nOutput the next function call you'd like to try in the requested format. If you have found a parameter set that converges, call the function with that parameter set with 'should_stop': True to avoid unecessary further calls."""
  
    
    return meta_prompt

def extract_string(raw, task=None):
    """
    Extract parameter value from LLM raw output using XML-like tags.
    
    Args:
        raw: Raw string output from LLM
        task: Task parameter name (e.g., "cfl", "n_space") for XML tag extraction
        
    Returns:
        dict or None: Extracted parameters as dictionary, or None if extraction fails
    """
    # First try to extract using XML-like tags if task is provided
    if task:
        start_tag = f"<{task}>"
        end_tag = f"</{task}>"
        
        if start_tag in raw:
            # Extract content between XML tags
            raw_after_start = raw[raw.index(start_tag) + len(start_tag):]
            if end_tag in raw_after_start:
                content = raw_after_start[:raw_after_start.index(end_tag)]
                
                # Try to parse the content as a number
                try:
                    # Clean up the content - remove extra whitespace, newlines
                    content = content.strip()
                    value = float(content)
                    return {"value": value}
                except ValueError:
                    pass
    
    # Fallback to the original extraction methods if XML tags don't work
    
    # Try to extract numerical values from the raw output
    # Look for standalone numbers (int or float)
    
    # First try to extract a simple number at the end or alone in the string
    number_patterns = [
        r'([+-]?\d*\.?\d+)\s*$',  # Number at end of string
        r'^([+-]?\d*\.?\d+)\s*$',  # Number as entire string
        r'([+-]?\d*\.?\d+)(?:\s|$)',  # Number followed by whitespace or end
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, raw.strip())
        if match:
            try:
                value_str = match.group(1)
                # Try to convert to appropriate type
                if '.' in value_str:
                    value = float(value_str)
                else:
                    # Could be int or float, let's default to float for safety
                    value = float(value_str)
                
                return {"value": value}
                
            except (ValueError, IndexError):
                continue
    
    # If no simple number found, try to extract from more complex text
    # Look for patterns like "I suggest 0.5" or "The optimal value is 0.75"
    suggestion_patterns = [
        r'(?:suggest|recommend|propose)(?:\s+(?:a\s+)?(?:value\s+)?(?:of\s+)?)?([+-]?\d*\.?\d+)',
        r'(?:optimal|best)(?:\s+value)?(?:\s+is)?(?:\s+)?([+-]?\d*\.?\d+)',
        r'(?:answer|result)(?:\s+is)?(?:\s+)?([+-]?\d*\.?\d+)',
        r'(?:should\s+be|would\s+be)(?:\s+)?([+-]?\d*\.?\d+)',
    ]
    
    raw_lower = raw.lower()
    for pattern in suggestion_patterns:
        match = re.search(pattern, raw_lower)
        if match:
            try:
                value = float(match.group(1))
                return {"value": value}
            except (ValueError, IndexError):
                continue
    
    # If still nothing found, look for any number in the text
    all_numbers = re.findall(r'([+-]?\d*\.?\d+)', raw)
    if all_numbers:
        try:
            # Take the last number found (often the conclusion)
            value = float(all_numbers[-1])
            return {"value": value}
        except ValueError:
            pass
    
    return None

def get_dummy(qid, verifier):
    """
    Get dummy/ground truth solution for comparison.
    
    Args:
        profile: Profile identifier
        problem: Problem type
        task: Task parameter
        
    Returns:
        dict: Ground truth parameter values
    """
    return verifier.best_params[qid]