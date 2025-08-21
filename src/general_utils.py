import sys, json, yaml, os
import numpy as np
sys.path.append("/home/ubuntu/dev/SimulCost-Bench")
import costsci_tools.wrappers as wrappers
import pdb

NAME_TO_FOLDER = {
    "1D_heat_transfer":"heat_1d",
    "2D_heat_transfer":"heat_steady_2d",
    "1D_burgers":"burgers_1d",
    "euler_1d": "euler_1d"
}

_PROBLEM_TO_TOOL_NAME = {
    "1D_heat_transfer":{
        "cfl":"heat_1d_check_converge_cfl",
        "n_space":"heat_1d_check_converge_n_space"
    },
    "2D_heat_transfer":{
        
    },
    "1D_burgers":{
        
    }
}

def params_to_tool_args(problem, dict):
    response = {}
    if problem == "1D_heat_transfer":
        response["current_cfl"] = dict["cfl"]
        response["current_n_space"] = dict["n_space"]
    elif problem == "2D_heat_transfer":
        response["current_dx"] = dict["dx"]
    
    return response
        
INFO = {
    "cfl":{
        "description":"cfl, the number of Courant-Friedrichs-Lewy condition which establishes a relationship between temporal and spatial discretization, to solve a given PDE problem.",
        "hints":"Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.\nPlease strike a balance between being too conservative and too aggressive:\n- If cfl is too large, the process may fail to converge.\n- If it's too small, the cost may increase dramatically.",
        "range": [0.01, 1.01],
        "initial": 1.0,
        "type": "float",
        "search": "dec",
        "best_key": "best_CFL" 
    },
    "n_space":{
        "description":"n_space, the number of spatial segments to solve a given PDE problem.",
        "hints":"Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.\nPlease strike a balance between being too conservative and too aggressive:\n- If n_space is too small, the process may fail to converge.\n- If it's too large, the cost may increase dramatically.", 
        "range": [64, 1000],
        "initial": 100,
        "type": "int",
        "search": "grid",
        "best_key": "best_n_space"
    },
    "dx":{
        "range":[0, 1.01],
        "initial": 0.005,
        "best_key": "best_dx"
    },
    "error_threshold":{
        "range": [-1, 1.01],
        "initial": 1e-7,
        "best_key": "best_error_threshold"
    },
    "relax":{
        "range":[0, 1.01],
        "initial": 1.0,
        "best_key": "optimal_relaxation_factor"
    },
    "t_init":{
        "range": [0, 1.01],
        "initial": 0.25,
        "best_key": "optimal_initial_temperature"
    },
    "w":{
        "range": [1.0, 2.0],
        "initial": 1.0,
        "best_key": "best_w"
    },
    "k":{
        "range": [-1, 1],
        "initial": 1,
        "best_key": "best_k"
    }
}

def soft_success(d, epsilon):
    """计算单个 (d, epsilon) 对的 Soft Success 值"""
    r = d / epsilon
    
    if r <= 1:
        return 1.0
    
    # 参数
    alpha = 0.6
    beta = 0.43
    gamma = 1.5
    omega = 0.3
    delta = 2.2
    
    # 双组分衰减函数
    exp_component = np.exp(-beta * (r - 1)**gamma)
    logistic_component = 1 / (1 + omega * (r - 1)**delta)
    
    return alpha * exp_component + (1 - alpha) * logistic_component

def check_cost(problem, profile, params):
    runner = getattr(wrappers, f"run_sim_{NAME_TO_FOLDER[problem]}")
    if problem == "1D_heat_transfer":
        cost = runner(profile, params['cfl'], int(params['n_space']))
    elif problem == "2D_heat_transfer":
        cost, steps = runner(profile, params["dx"], params["relax"], params["error_threshold"], params["t_init"]) 
    elif problem == "euler_1d":
        cost = runner(profile, params["cfl"], params["beta"], params["k"], params["n_space"])

    return cost

def check_gt(
    problem,
    profile,
    gt,
    x,
    tolerance=0.05,
    whether_soft_success=True
):
    compare_func = getattr(wrappers, f"compare_res_{NAME_TO_FOLDER[problem]}")
    if problem == "1D_heat_transfer":
        success, error = compare_func(profile, x['cfl'], int(x['n_space']),
                                      profile, gt['cfl'], int(gt['n_space']), tolerance)
    elif problem == "2D_heat_transfer":
        success, error = compare_func(profile, x["dx"], x["relax"], x["error_threshold"], x["t_init"],
                                  profile, gt["dx"], gt["relax"], gt["error_threshold"], gt["t_init"], tolerance)
    elif problem == "1D_burgers":
        success, error = compare_func(profile, x["cfl"], x["k"], x["w"],
                                  profile, gt["cfl"], gt["k"], gt["w"],
                                  tolerance[0], tolerance[1])
    elif problem == "euler_1d":
        success, success1, success2, error = compare_func(profile, x["cfl"], x["beta"], x["k"],
                                      profile, gt["cfl"], gt["beta"], gt["k"],
                                      tolerance, x["n_space"], gt["n_space"])
    if whether_soft_success:
        return soft_success(error, tolerance)
    else:
        return error < tolerance

def extract_yaml_parameter(yaml_path, parameter_name, default_value=None):
    """
    Extract a specific parameter from a YAML configuration file.
    
    Args:
        yaml_path (str): Path to the YAML file
        parameter_name (str): Name of the parameter to extract
        default_value: Default value to return if parameter not found
        
    Returns:
        The parameter value if found, otherwise default_value
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if config is None:
            return default_value
            
        # Support nested parameter access using dot notation (e.g., "section.param")
        if '.' in parameter_name:
            keys = parameter_name.split('.')
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default_value
            return value
        else:
            return config.get(parameter_name, default_value)
            
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")
    

def extract_static(profile_path):
    """
    Extract static (non-tunable) parameters from a YAML profile file.
    
    Args:
        profile_path (str): Path to the YAML profile file
        
    Returns:
        dict: Dictionary containing static parameters
    """
    config = yaml.safe_load(open(profile_path, 'r'))
    
    # Define which parameters are static for each problem type
    # These parameters don't get optimized and remain constant
    static_params = {}
    
    # Problem-specific static parameters
    heat_1d_static = ['L', 'k', 'h', 'rho', 'cp', 'T_inf', 'T_init']
    #heat_2d_static = ['T_top', 'T_bottom', 'T_left', 'T_right']  
    #burgers_1d_static = ['L', 'case']
    euler_1d_static = ['L', 'gamma', 'case']
    
    # Extract all parameters that aren't tunable
    all_static_keys = set()
    
    # Add problem-specific static parameters based on what's in the config
    if any(key in config for key in heat_1d_static):
        all_static_keys.update(heat_1d_static)
    if any(key in config for key in euler_1d_static):
        all_static_keys.update(euler_1d_static)
    
    # Extract static parameters that exist in the config
    for key in all_static_keys:
        if key in config:
            static_params[key] = config[key]
            
    return static_params

def complete_params_with_initial(params, problem):
    """
    Complete a parameter dictionary with initial values from INFO for parameters not specified.
    
    Args:
        params (dict): Dictionary containing only the to-be-decided parameter(s)
        problem (str): Problem type ("1D_heat_transfer", "2D_heat_transfer", "1D_burgers")
        
    Returns:
        dict: Complete parameter dictionary with initial values filled in
    """
    completed_params = params.copy()
    
    # Define which parameters are required for each problem type
    if problem == "1D_heat_transfer":
        required_params = ["cfl", "n_space"]
    elif problem == "2D_heat_transfer":
        required_params = ["dx", "relax", "error_threshold", "t_init"]
    elif problem == "1D_burgers":
        required_params = ["cfl", "k", "w"]
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    
    # Fill in missing parameters with their initial values
    for param in required_params:
        if param not in completed_params:
            if param in INFO and "initial" in INFO[param]:
                completed_params[param] = INFO[param]["initial"]
            else:
                # Fallback to range midpoint if no initial value specified
                if param in INFO and "range" in INFO[param]:
                    param_range = INFO[param]["range"]
                    completed_params[param] = (param_range[0] + param_range[1]) / 2
                else:
                    raise ValueError(f"No initial value or range found for parameter: {param}")
    
    return completed_params

def get_tolerance(problem, level):
    '''
    problem: 'heat_1d', 'burgers_1d', 'euler_1d', 'ns_channel_2d'
    level: 'low', 'medium', 'high'
    
    returns a float or dict depending on problem type
    '''
    tolerance_map = {
        '1D_heat_transfer': {
            'low': 0.01,
            'medium': 0.001,
            'high': 0.0001
        },
        'burgers_1d': {
            'low': {'tolerance_rmse': 0.01, 'tolerance_linf': 0.05},
            'medium': {'tolerance_rmse': 0.005, 'tolerance_linf': 0.025},
            'high': {'tolerance_rmse': 0.001, 'tolerance_linf': 0.01}
        },
        'euler_1d': {
            'low': 0.08,
            'medium': 0.02,
            'high': 0.01
        },
        'ns_channel_2d': {
            'low': {
                'mass_tolerance': 0.0005,
                'u_rmse_tolerance': 0.05,
                'v_rmse_tolerance': 0.05,
                'p_rmse_tolerance': 0.05
            },
            'medium': {
                'mass_tolerance': 0.0001,
                'u_rmse_tolerance': 0.03,
                'v_rmse_tolerance': 0.03,
                'p_rmse_tolerance': 0.03
            },
            'high': {
                'mass_tolerance': 5.0e-05,
                'u_rmse_tolerance': 0.01,
                'v_rmse_tolerance': 0.01,
                'p_rmse_tolerance': 0.01
            }
        }
    }
    
    if problem not in tolerance_map:
        raise ValueError(f"Unknown problem: {problem}. Supported problems: {list(tolerance_map.keys())}")
    
    if level not in tolerance_map[problem]:
        raise ValueError(f"Unknown level: {level}. Supported levels: {list(tolerance_map[problem].keys())}")
    
    return tolerance_map[problem][level]

class Verifier:
    def __init__(self,
                 problem,
                 task,
                 tolerance,
                 dummy_root="/home/ubuntu/dev/SimulCost-Bench/data",
                 profile_root="/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs"):
        self.problem = problem
        self.task = task
        self.tolerance = tolerance
        
        dummy_path = f"{dummy_root}/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/zero_shot_questions.json"
        #dummy_path = f"{dummy_root}/{task}/zero_shot_question.json"
        with open(dummy_path, 'r') as f:
            data = json.load(f)
        #self.best_params = {x["profile"]: self.get_best_params(x) for x in data}
        self.best_params = {x["profile"]: x["best_params"] for x in data}
        self.best_costs = {x["profile"]: x["dummy_cost"] for x in data}
        
        self.profile_root = profile_root
        
    def get_best_params(self, x):
        for y in x["param_history"]:
            if y[self.task] == x[INFO[self.task]["best_key"]]:
                return y
        raise ValueError("Verifier initialization error: no dummy best sol found.")
    
    def metric(self, param, profile, tolerance=None, soft_success=True):
        if not tolerance:
            if self.problem in ["1D_burgers"]:
                tolerance = [
                    extract_yaml_parameter(f"{self.profile_root}/{NAME_TO_FOLDER[self.problem]}/{profile}.yaml",
                                            "tolerance1",
                                            1e-2),
                    extract_yaml_parameter(f"{self.profile_root}/{NAME_TO_FOLDER[self.problem]}/{profile}.yaml",
                                            "tolerance2",
                                            1e-3),
                ]
            else:
                tolerance = extract_yaml_parameter(f"{self.profile_root}/{NAME_TO_FOLDER[self.problem]}/{profile}.yaml",
                                            "tolerance",
                                            1e-4)
        #success = check_gt(self.problem, profile, self.best_params[profile], param, float(tolerance), profile_root=self.profile_root)
        success = check_gt(self.problem,
                           profile,
                           self.best_params[profile],
                           param,
                           get_tolerance(self.problem, self.tolerance),
                           whether_soft_success=soft_success)
        cost = check_cost(self.problem, profile, param)
        score = success * (self.best_costs[profile] / (1e-3 + cost))
        return success, cost, score
        
