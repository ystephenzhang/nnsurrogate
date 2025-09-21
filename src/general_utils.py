import sys, json, yaml, os
import numpy as np
sys.path.append("/home/ubuntu/dev/SimulCost-Bench")
import costsci_tools.wrappers as wrappers
import pdb
import torch

NUMERICAL_PROMPT = """
Your task is to optimize a two-dimensional black-box function with a given parameter. You will be prompted with a list of history of parameter and values, where values include an accuracy indicator and success indicator. You are required to first optimize accuracy until it reaches 1.0, then optimize efficiency for as high as possible. The parameter in history will start with <n_space> and end with </n_space>. Please return a parameter value different from all values given in the history that you think will optimize the function value as requested. Please return your answer by starting with <task> and ending with </task> as well. **You may NOT use any form of prior knowledge, and treat all parameter names, function names, etc. as purely arbitrary.**
"""
NUMERICAL_PROMPT_ITERATIVE = """
Your task is to minimize a black-box function's value by refining n_space value as input to the function, until you are prompted with "converged". Many metrics will be returned to you, but you are to minimize "RMSE" alone. \nWorkflow:\nStep 1: Estimate an initial fairly coarse choice of n_space, as you will gradually refine the solution and check convergence.\nStep 2: Call the Convergence Test Function; check if converged.\nStep 3: Refine n_space based on the feedback from the simulation.\nStep 4: You have at most 10 total opportunities to refine your resolution.Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. Please response in the form {{"n_space": x, "should_stop": y}}. **You may NOT use any form of prior knowledge, and treat all parameter names, function names, etc. as purely arbitrary.**
"""

NAME_TO_FOLDER = {
    "1D_heat_transfer":"heat_1d",
    "heat_1d":"heat_1d",
    "2D_heat_transfer":"heat_steady_2d",
    "1D_burgers":"burgers_1d",
    "burgers_1d":"burgers_1d",
    "euler_1d": "euler_1d",
    "ns_channel_2d": "ns_channel_2d",
    "ns_transient_2d": "ns_transient_2d",
    "epoch": "epoch"
}

_PROBLEM_TO_TOOL_NAME = {
    "heat_1d":{
        "cfl":"heat_1d_check_converge_cfl",
        "n_space":"heat_1d_check_converge_n_space"
    },
    "2D_heat_transfer":{
        
    },
    "1D_burgers":{
        
    },
    "euler_1d":{
        "cfl":"euler_1d_check_converge_cfl",
        "beta":"euler_1d_check_converge_beta",
        "k":"euler_1d_check_converge_k",
        "n_space":"euler_1d_check_converge_n_space"
    },
    "ns_channel_2d":{
        "mesh_x":"ns_channel_2d_check_converge_mesh_x",
        "mesh_y":"ns_channel_2d_check_converge_mesh_y",
        "omega_u":"ns_channel_2d_check_converge_omega_u",
        "omega_v":"ns_channel_2d_check_converge_omega_v",
        "omega_p":"ns_channel_2d_check_converge_omega_p",
        "diff_u_threshold":"ns_channel_2d_check_converge_diff_u_threshold",
        "diff_v_threshold":"ns_channel_2d_check_converge_diff_v_threshold",
        "res_iter_v_threshold":"ns_channel_2d_check_converge_res_iter_v_threshold"
    },
    "ns_transient_2d":{
        "resolution":"ns_transient_2d_check_converge_resolution" 
    }
}

def params_to_tool_args(problem, dict):
    response = {}
    if problem == "heat_1d":
        response["current_cfl"] = dict["cfl"]
        response["current_n_space"] = dict["n_space"]
    elif problem == "2D_heat_transfer":
        response["current_dx"] = dict["dx"]
    elif problem == "ns_channel_2d":
        response["current_mesh_x"] = dict["mesh_x"]
        response["current_mesh_y"] = dict["mesh_y"]
        response["current_omega_u"] = dict["omega_u"]
        response["current_omega_v"] = dict["omega_v"]
        response["current_omega_p"] = dict["omega_p"]
        response["current_diff_u_threshold"] = dict["diff_u_threshold"]
        response["current_diff_v_threshold"] = dict["diff_v_threshold"]
        response["current_res_iter_v_threshold"] = dict["res_iter_v_threshold"]
    elif problem == "ns_transient_2d":
        response["current_resolution"] = dict["resolution"]
        response["current_cfl"] = dict["cfl"]
        response["current_relaxation_factor"] = dict["relaxation_factor"]
        response["current_residual_threshold"] = dict["residual_threshold"]
    
    return response
        
INFO = {
    "resolution":{
        "description": "Spatial grid resolution - controls mesh density for the simulation",
        "range": [50, 256],
        "type": "int",
        "best_key": "best_resolution"
    },
    "mesh_x":{
        "range": [16, 128],
        #"range": [32, 256],
    },
    "mesh_y":{
        "range": [16, 64]
    },
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
        "range": [40, 2048],
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
    },
    "omega_u":{
        "description": "Under-relaxation factor for u-velocity - controls convergence rate and stability",
        "range": [0.1, 1.0],
        "initial": 0.6,
        "type": "float",
        "search": "grid",
        "best_key": "best_omega_u"
    },
    "omega_v":{
        "description": "Under-relaxation factor for v-velocity - controls convergence rate and stability", 
        "range": [0.1, 1.0],
        "initial": 0.6,
        "type": "float",
        "search": "grid",
        "best_key": "best_omega_v"
    },
    "omega_p":{
        "description": "Under-relaxation factor for pressure - controls pressure correction convergence",
        "range": [0.1, 0.5],
        "initial": 0.3,
        "type": "float",
        "search": "grid",
        "best_key": "best_omega_p"
    },
    "diff_u_threshold":{
        "description": "Convergence threshold for u-velocity difference between iterations",
        "range": [1e-07, 1e-03],
        "initial": 1e-07,
        "type": "float",
        "search": "grid",
        "best_key": "best_diff_u_threshold"
    },
    "diff_v_threshold":{
        "description": "Convergence threshold for v-velocity difference between iterations",
        "range": [1e-07, 1e-03],
        "initial": 1e-07,
        "type": "float",
        "search": "grid",
        "best_key": "best_diff_v_threshold"
    },
    "res_iter_v_threshold":{
        "description": "Residual threshold for inner v-velocity iterations - can be fixed value or exp_decay",
        "range": [1e-07, 1e-03],
        "initial": "exp_decay",
        "type": "mixed",
        "search": "grid",
        "best_key": "best_res_iter_v_threshold"
    },
    "relaxation_factor":{
        "description": "Relaxation factor for iterative solver - controls convergence rate",
        "range": [1.0, 2.0],
        "initial": 1.3,
        "type": "float",
        "search": "grid",
        "best_key": "best_relaxation_factor"
    },
    "residual_threshold":{
        "description": "Residual threshold for convergence - controls solution accuracy",
        "range": [1e-05, 1e-01],
        "initial": 1e-02,
        "type": "float",
        "search": "grid",
        "best_key": "best_residual_threshold"
    },
    "nx":{
        "description": "Number of grid cells for spatial discretization - determines spatial resolution",
        "range": [200, 3200],
        "initial": 400,
        "type": "int",
        "search": "grid",
        "best_key": "best_nx"
    },
    "dt_multiplier":{
        "description": "Time-increment multiplier - controls temporal resolution",
        "range": [0.80, 0.99],
        "initial": 0.95,
        "type": "float",
        "search": "grid",
        "best_key": "best_dt_multiplier"
    },
    "npart":{
        "description": "Number of pseudoparticles per cell",
        "range": [10, 50],
        "initial": 20,
        "type": "int",
        "search": "grid",
        "best_key": "best_npart"
    },
    "field_order":{
        "description": "Field integration order",
        "range": [2, 6],
        "initial": 2,
        "type": "int",
        "search": "discrete",
        "values": [2, 4, 6],
        "best_key": "best_field_order"
    },
    "particle_order":{
        "description": "Particle weighting order",
        "range": [2, 5],
        "initial": 3,
        "type": "int",
        "search": "discrete",
        "values": [2, 3, 5],
        "best_key": "best_particle_order"
    }
}

def reverse_preproc(cost, params):
    if "n_space" in params:
        n_space = params["n_space"]
        cfl = params["cfl"]
        cost *= n_space * n_space * n_space / cfl
    return cost

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

def soft_success_multi(d_list, epsilon_list):
    """Calculate average Soft Success value for multiple (d, epsilon) pairs"""
    ss_values = []
    for d, eps in zip(d_list, epsilon_list):
        ss = soft_success(d, eps)
        ss_values.append(ss)
    
    return np.mean(ss_values)  # Arithmetic mean

def check_cost(problem, profile, params):
    runner = getattr(wrappers, f"run_sim_{NAME_TO_FOLDER[problem]}")
    if problem == "heat_1d" or problem == "1D_heat_transfer":
        cost = runner(profile, params['cfl'], int(params['n_space']))
    elif problem == "2D_heat_transfer":
        cost, steps = runner(profile, params["dx"], params["relax"], params["error_threshold"], params["t_init"]) 
    elif problem == "burgers_1d":
        # Use beta parameter but pass as w to the wrapper function
        cost = runner(profile, params["cfl"], params["k"], params["beta"], params["n_space"])
    elif problem == "euler_1d":
        cost = runner(profile, params["cfl"], params["beta"], params["k"], params["n_space"])
    elif problem == "ns_channel_2d":
        profile_to_bt = {
            "p1": "channel_flow",
            "p2": "back_stair_flow",
            "p3": "expansion_channel",
            "p4": "cube_driven_flow"
        }
        cost, steps = runner(profile, profile_to_bt[profile] , int(params["mesh_x"]), int(params["mesh_y"]), 
                     params["omega_u"], params["omega_v"], params["omega_p"],
                     params["diff_u_threshold"], params["diff_v_threshold"], 
                     params["res_iter_v_threshold"])
    elif problem == "ns_transient_2d":
        # Extract static parameters from profile configuration
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        boundary_condition = extract_yaml_parameter(profile_path, 'boundary_condition', 1)
        reynolds_num = extract_yaml_parameter(profile_path, 'reynolds_num', 1000.0)
        total_runtime = extract_yaml_parameter(profile_path, 'total_runtime', 1.0)

        # Prepare other parameters that are not in the main function signature
        other_params = {}
        advection_scheme = extract_yaml_parameter(profile_path, 'advection_scheme', 'cip')
        vorticity_confinement = extract_yaml_parameter(profile_path, 'vorticity_confinement', 0.0)
        no_dye = extract_yaml_parameter(profile_path, 'no_dye', False)
        cpu = extract_yaml_parameter(profile_path, 'cpu', True)
        visualization = extract_yaml_parameter(profile_path, 'visualization', 0)

        if advection_scheme != 'cip':
            other_params['advection_scheme'] = advection_scheme
        if vorticity_confinement != 0.0:
            other_params['vorticity_confinement'] = vorticity_confinement
        if no_dye != False:
            other_params['no_dye'] = no_dye
        if cpu != True:
            other_params['cpu'] = cpu
        if visualization != 0:
            other_params['visualization'] = visualization

        cost, steps = runner(profile, boundary_condition, int(params["resolution"]),
                             reynolds_num, params["cfl"], params["relaxation_factor"],
                             params["residual_threshold"], total_runtime, other_params)
    elif problem == "epoch":
        runner = getattr(wrappers, f"runEpoch")
        cost = runner(profile, int(params["nx"]), params["dt_multiplier"],
                     int(params["npart"]), int(params["field_order"]),
                     int(params["particle_order"]))

    return cost

def check_gt(
    problem,
    profile,
    gt,
    x,
    tolerance="medium",
    whether_soft_success=True
):
    compare_func = getattr(wrappers, f"compare_res_{NAME_TO_FOLDER[problem]}")
    
    if problem == "heat_1d" or problem == "1D_heat_transfer":
        # Map tolerance level to numerical value for heat_1d
        tolerance_map = {
            'low': 0.01,
            'medium': 0.001,
            'high': 0.0001
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.001)
        success, error = compare_func(profile, x['cfl'], int(x['n_space']),
                                      profile, gt['cfl'], int(gt['n_space']), numeric_tolerance)
    elif problem == "2D_heat_transfer":
        # Map tolerance level to numerical value for 2D_heat_transfer
        tolerance_map = {
            'low': 0.1,
            'medium': 0.01,
            'high': 0.001
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.01)
        success, error = compare_func(profile, x["dx"], x["relax"], x["error_threshold"], x["t_init"],
                                  profile, gt["dx"], gt["relax"], gt["error_threshold"], gt["t_init"], numeric_tolerance)
    elif problem == "burgers_1d":
        # Map tolerance level to numerical value for burgers_1d
        tolerance_map = {
            'low': 0.08,
            'medium': 0.04,
            'high': 0.01
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.04)
        success, _, _, error = compare_func(profile, x["cfl"], x["k"], x["beta"],
                                  profile, gt["cfl"], gt["k"], gt["beta"],
                                  numeric_tolerance,
                                  x["n_space"], gt["n_space"])
    elif problem == "euler_1d":
        # Map tolerance level to numerical value for euler_1d
        tolerance_map = {
            'low': 0.08,
            'medium': 0.02,
            'high': 0.01
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.02)
        success, success1, success2, error = compare_func(profile, x["cfl"], x["beta"], x["k"],
                                      profile, gt["cfl"], gt["beta"], gt["k"],
                                      numeric_tolerance, x["n_space"], gt["n_space"])
    elif problem == "ns_channel_2d":
        profile_to_bt = {
            "p1": "channel_flow",
            "p2": "back_stair_flow", 
            "p3": "expansion_channel",
            "p4": "cube_driven_flow"
        }
        boundary_type = profile_to_bt[profile]
        
        # Map tolerance level to tolerance dictionary for ns_channel_2d
        tolerance_map = {
            'low': {
                'mass_tolerance': 1e-04,
                'u_rmse_tolerance': 0.11,
                'v_rmse_tolerance': 0.05,
                'p_rmse_tolerance': 0.4
            },
            'medium': {
                'mass_tolerance': 1e-06,
                'u_rmse_tolerance': 0.02,
                'v_rmse_tolerance': 0.005,
                'p_rmse_tolerance': 0.2
            },
            'high': {
                'mass_tolerance': 1e-08,
                'u_rmse_tolerance': 0.008,
                'v_rmse_tolerance': 0.001,
                'p_rmse_tolerance': 0.10
            }
        }
        tolerance_dict = tolerance_map.get(tolerance, tolerance_map['medium'])
        
        # Read length and breadth from profile
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        length = extract_yaml_parameter(profile_path, 'length', 20.0)
        breadth = extract_yaml_parameter(profile_path, 'breadth', 1.0)
        
        success, rmse_u, rmse_v, rmse_p, mass_conserved1, mass_conserved2 = compare_func(
            profile1=profile,
            boundary_type1=boundary_type,
            mesh_x1=int(x["mesh_x"]),
            mesh_y1=int(x["mesh_y"]),
            omega_u1=x["omega_u"],
            omega_v1=x["omega_v"],
            omega_p1=x["omega_p"],
            diff_u_threshold1=x["diff_u_threshold"],
            diff_v_threshold1=x["diff_v_threshold"],
            res_iter_v_threshold1=x["res_iter_v_threshold"],
            profile2=profile,
            boundary_type2=boundary_type,
            mesh_x2=int(gt["mesh_x"]),
            mesh_y2=int(gt["mesh_y"]),
            omega_u2=gt["omega_u"],
            omega_v2=gt["omega_v"],
            omega_p2=gt["omega_p"],
            diff_u_threshold2=gt["diff_u_threshold"],
            diff_v_threshold2=gt["diff_v_threshold"],
            res_iter_v_threshold2=gt["res_iter_v_threshold"],
            length=length,
            breadth=breadth,
            mass_tolerance=tolerance_dict['mass_tolerance'],
            u_rmse_tolerance=tolerance_dict['u_rmse_tolerance'],
            v_rmse_tolerance=tolerance_dict['v_rmse_tolerance'],
            p_rmse_tolerance=tolerance_dict['p_rmse_tolerance']
        )
        # Calculate combined error metric (average of RMSEs)
        error = (rmse_u + rmse_v + rmse_p) / 3.0
        
    elif problem == "ns_transient_2d":
        # Map tolerance level to numerical value for ns_transient_2d
        tolerance_map = {
            'low': 0.6,
            'medium': 0.3,
            'high': 0.15
        }
        numeric_tolerance = tolerance_map.get(tolerance)
        
        # Extract static parameters from profile configuration for both x and gt
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        boundary_condition = extract_yaml_parameter(profile_path, 'boundary_condition', 1)
        reynolds_num = extract_yaml_parameter(profile_path, 'reynolds_num', 1000.0)
        total_runtime = extract_yaml_parameter(profile_path, 'total_runtime', 1.0)

        # Prepare other parameters that are not in the main function signature
        other_params = {}
        advection_scheme = extract_yaml_parameter(profile_path, 'advection_scheme', 'cip')
        vorticity_confinement = extract_yaml_parameter(profile_path, 'vorticity_confinement', 0.0)
        no_dye = extract_yaml_parameter(profile_path, 'no_dye', False)
        cpu = extract_yaml_parameter(profile_path, 'cpu', True)
        visualization = extract_yaml_parameter(profile_path, 'visualization', 0)

        if advection_scheme != 'cip':
            other_params['advection_scheme'] = advection_scheme
        if vorticity_confinement != 0.0:
            other_params['vorticity_confinement'] = vorticity_confinement
        if no_dye != False:
            other_params['no_dye'] = no_dye
        if cpu != True:
            other_params['cpu'] = cpu
        if visualization != 0:
            other_params['visualization'] = visualization
        #pdb.set_trace()
        success, error = compare_func(
            profile, boundary_condition, int(x["resolution"]), reynolds_num, x["cfl"],
            x["relaxation_factor"], x["residual_threshold"], total_runtime,
            profile, boundary_condition, int(gt["resolution"]), reynolds_num, gt["cfl"],
            gt["relaxation_factor"], gt["residual_threshold"], total_runtime,
            numeric_tolerance, other_params, other_params
        )
    elif problem == "epoch":
        # Map tolerance level to numerical value for epoch
        tolerance_map = {
            'low': 0.36,
            'medium': 0.33,
            'high': 0.30
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.33)
        success, error = compare_func(profile, int(x["nx"]), x["dt_multiplier"],
                                      int(x["npart"]), int(x["field_order"]), int(x["particle_order"]),
                                      profile, int(gt["nx"]), gt["dt_multiplier"],
                                      int(gt["npart"]), int(gt["field_order"]), int(gt["particle_order"]),
                                      numeric_tolerance)
        
    if whether_soft_success:
        # Use appropriate tolerance for soft_success calculation
        if problem == "ns_channel_2d":
            # Use soft_success_multi for multiple RMSE values
            rmse_list = [rmse_u, rmse_v, rmse_p]
            epsilon_list = [tolerance_dict['u_rmse_tolerance'], 
                           tolerance_dict['v_rmse_tolerance'], 
                           tolerance_dict['p_rmse_tolerance']]
            
            # Handle NaN/inf values in RMSE
            valid_pairs = []
            for rmse_val, eps_val in zip(rmse_list, epsilon_list):
                if not (np.isnan(rmse_val) or np.isinf(rmse_val)):
                    valid_pairs.append((rmse_val, eps_val))
            
            if valid_pairs:
                soft_success_value = soft_success_multi([r for r, e in valid_pairs], [e for r, e in valid_pairs])
            else:
                soft_success_value = 0.0
                
            return soft_success_value, error
        else:
            return soft_success(error, numeric_tolerance), error
    else:
        # For binary success, different logic for ns_channel_2d vs others
        if problem == "ns_channel_2d":
            # For ns_channel_2d, success is already determined by the comparison function
            return success, error
        elif problem == "ns_transient_2d":
            return success, error
        elif problem == "epoch":
            return success, error
        else:
            return error < numeric_tolerance, error

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
    burgers_1d_static = ['case']  # Only case for one-hot encoding
    euler_1d_static = ['case']
    ns_channel_2d_static = ['length', 'breadth', 'mu', 'rho', 'boundary_condition']
    ns_transient_2d_static = ['boundary_condition', 'reynolds_num']
    epoch_static = ['L', 'L_target', 'a0', 'laser_lambda', 'laser_time', 'n_target', 'end_time']

    # Extract all parameters that aren't tunable
    all_static_keys = set()

    # Add problem-specific static parameters based on what's in the config
    # Use more specific detection to avoid conflicts
    if "heat_1d" in profile_path:
        # Heat 1D specific - has unique thermal parameters
        all_static_keys.update(heat_1d_static)
    elif "euler_1d" in profile_path:
        # Euler 1D specific - has gamma parameter
        all_static_keys.update(euler_1d_static)
    elif "burgers_1d" in profile_path:
        # Burgers 1D specific - has w parameter (limiter parameter)
        all_static_keys.update(burgers_1d_static)
    elif "ns_channel_2d" in profile_path:
        # NS Channel 2D specific - has fluid properties and boundary conditions
        all_static_keys.update(ns_channel_2d_static)
    elif "ns_transient_2d" in profile_path:
        # NS Transient 2D specific - has boundary conditions and flow parameters
        all_static_keys.update(ns_transient_2d_static)
    elif "epoch" in profile_path:
        # Epoch specific - has laser and simulation parameters
        all_static_keys.update(epoch_static)
    
    
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
        problem (str): Problem type ("heat_1d", "2D_heat_transfer", "1D_burgers")
        
    Returns:
        dict: Complete parameter dictionary with initial values filled in
    """
    completed_params = params.copy()
    
    # Define which parameters are required for each problem type
    if problem == "heat_1d":
        required_params = ["cfl", "n_space"]
    elif problem == "2D_heat_transfer":
        required_params = ["dx", "relax", "error_threshold", "t_init"]
    elif problem == "1D_burgers":
        required_params = ["cfl", "k", "w"]
    elif problem == "ns_channel_2d":
        required_params = ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                          "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
    elif problem == "ns_transient_2d":
        required_params = ["resolution", "cfl", "relaxation_factor", "residual_threshold"]
    elif problem == "epoch":
        required_params = ["nx", "dt_multiplier", "npart", "field_order", "particle_order"]
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
    problem: 'heat_1d', 'burgers_1d', 'euler_1d', 'ns_channel_2d', 'ns_transient_2d'
    level: 'low', 'medium', 'high'
    
    returns a float or dict depending on problem type
    '''
    tolerance_map = {
        'heat_1d': {
            'low': 0.01,
            'medium': 0.001,
            'high': 0.0001
        },
        'burgers_1d': {
            'low': 0.08,
            'medium': 0.04,
            'high': 0.01,
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
        },
        'ns_transient_2d': {
            'low': 0.6,
            'medium': 0.3,
            'high': 0.15
        },
        'epoch': {
            'low': 0.36,
            'medium': 0.33,
            'high': 0.30
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
                 profile_root="/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs",
                 iterative=False):
        self.problem = problem
        self.task = task
        self.tolerance = tolerance
        self.iterative = iterative
        
        if not iterative:
            dummy_path = f"{dummy_root}/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/zero_shot_questions.json"
        else:
            dummy_path = f"{dummy_root}/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/iterative_questions.json"
        #dummy_path = f"{dummy_root}/{task}/zero_shot_question.json"
        with open(dummy_path, 'r') as f:
            data = json.load(f)
        #self.best_params = {x["profile"]: self.get_best_params(x) for x in data}
        self.best_params = {str(x["QID"]): x["best_params"] for x in data}
        self.best_costs = {str(x["QID"]): x["dummy_cost"] for x in data}
        
        self.profile_root = profile_root
        
    def get_best_params(self, x):
        for y in x["param_history"]:
            if y[self.task] == x[INFO[self.task]["best_key"]]:
                return y
        raise ValueError("Verifier initialization error: no dummy best sol found.")
    
    def metric(self, param, profile, qid, soft_success=True, prev_cost=0):
        success, error = check_gt(self.problem,
                           profile,
                           self.best_params[qid],
                           param,
                           self.tolerance,
                           whether_soft_success=soft_success)
        cost = check_cost(self.problem, profile, param)
        if self.iterative:
            cost += prev_cost
        score = success * (self.best_costs[qid] / (1e-3 + cost))
        #efficiency = self.best_costs[profile] / (1e-3 + cost)
        return success, cost, score
        
    def raw_metric(self,
                   param,
                   profile,
                   qid,
                   prev_cost=0
                   ):
        success, error = check_gt(self.problem,
                           profile,
                           self.best_params[qid],
                           param,
                           self.tolerance,
                           whether_soft_success=True)
        cost = check_cost(self.problem, profile, param)
        if self.iterative:
            cost += prev_cost
        return error, success, cost, self.best_costs[qid] / (1e-3 + cost)

# Mapper functions for converting parameters to tensors
def heat_1d_static_mapper(profile_path, level):
    """Map static parameters for heat_1d to tensor format"""
    import torch
    static_params = extract_static(profile_path)
    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level, 1)  # Default to medium
    return torch.tensor([
        static_params.get('L', 1.0),
        static_params.get('k', 1.0), 
        static_params.get('h', 1.0),
        static_params.get('rho', 1.0),
        static_params.get('cp', 1.0),
        static_params.get('T_inf', 300.0),
        static_params.get('T_init', 350.0),
        level_encoding
    ], dtype=torch.float32)

def heat_1d_tunable_mapper(params):
    """Map tunable parameters for heat_1d to tensor format"""
    return torch.tensor([
        params.get('cfl', 1.0),
        params.get('n_space', 100)
    ])

def burgers_1d_static_mapper(profile_path, level):
    """Map static parameters for burgers_1d to tensor format"""
    import torch
    static_params = extract_static(profile_path)
    case = static_params.get('case')
    oh_mapping = {
        "sin":0,
        "rarefaction":1,
        "sod":2,
        "double_shock":3,
        "blast":4
    }
    # One-hot encoding for case parameter
    case_onehot = torch.zeros(5)
    case_onehot[oh_mapping[case]] = 1
    
    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level, 1)  # Default to medium
    return torch.concatenate([case_onehot, torch.tensor([level_encoding])], dim=-1)

def burgers_1d_tunable_mapper(params):
    """Map tunable parameters for burgers_1d to tensor format"""
    return torch.tensor([
        params.get('cfl', 0.5),
        params.get('n_space', 100),
        params.get('beta', 1.0),
        params.get('k', 1.0),  # Using beta instead of w
    ])

def euler_1d_static_mapper(profile_path, level):
    """Map static parameters for euler_1d to tensor format"""
    import torch
    static_params = extract_static(profile_path)
    case = static_params.get('case')

    case_mapping = {
        "sod": 0,
        "lax": 1,
        "mach_3": 2
    }
    
    case_onehot = torch.zeros(3)
    case_onehot[case_mapping[case]] = 1
    
    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level, 1)  # Default to medium
    return torch.concatenate([
        case_onehot,
        torch.tensor([level_encoding])
    ])

def euler_1d_tunable_mapper(params):
    """Map tunable parameters for euler_1d to tensor format"""
    return torch.tensor([
        params.get('cfl', 0.5),
        params.get('n_space', 100),
        params.get('beta', 1.0),
        params.get('k', 1.0)
    ])

def heat_1d_preprocessor(tunable_tensor, result_tensor, preproc_success=False, preproc_cost=False):
    """Preprocess result tensor for heat_1d"""
    processed = result_tensor.clone()
    if preproc_cost:
        n_space = tunable_tensor[1]
        cfl = tunable_tensor[0]
        processed[2] = result_tensor[2] * cfl / (n_space * n_space * n_space)
    return processed

def burgers_1d_preprocessor(tunable_tensor, result_tensor, preproc_success=False, preproc_cost=False):
    """Preprocess result tensor for burgers_1d"""
    processed = result_tensor.clone()
    if preproc_cost:
        n_space = tunable_tensor[1]
        cfl = tunable_tensor[0]
        processed[1] = result_tensor[1] * cfl / (n_space * n_space)
    return processed

def euler_1d_preprocessor(tunable_tensor, result_tensor, preproc_success=False, preproc_cost=False):
    """Preprocess result tensor for euler_1d"""
    processed = result_tensor.clone()
    if preproc_cost:
        n_space = tunable_tensor[1]
        cfl = tunable_tensor[0]
        processed[1] = result_tensor[1] * cfl / (n_space * n_space)
    return processed

def ns_channel_2d_static_mapper(profile_path, level):
    """Map static parameters for ns_channel_2d to tensor format"""
    import torch
    static_params = extract_static(profile_path)
    
    # One-hot encoding for boundary condition
    boundary_condition = static_params.get('boundary_condition', 'channel_flow')
    boundary_conditions = ['channel_flow', 'back_stair_flow', 'expansion_channel', 'cube_driven_flow']
    boundary_onehot = torch.zeros(len(boundary_conditions))
    if boundary_condition in boundary_conditions:
        boundary_onehot[boundary_conditions.index(boundary_condition)] = 1
    
    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level)  # Default to medium
    
    # Static fluid properties and parameters
    static_tensor = torch.tensor([
        static_params.get('length', 20.0),
        static_params.get('breadth', 1.0),
        static_params.get('mu', 0.01),
        static_params.get('rho', 1.0),
        level_encoding
    ], dtype=torch.float32)
    
    return torch.concatenate([static_tensor, boundary_onehot], dim=-1)

def ns_channel_2d_tunable_mapper(params):
    """Map tunable parameters for ns_channel_2d to tensor format"""
    # Handle res_iter_v_threshold which can be string or float
    res_iter_v_threshold = params.get('res_iter_v_threshold', 'exp_decay')
    if isinstance(res_iter_v_threshold, str) and res_iter_v_threshold == 'exp_decay':
        res_iter_v_threshold_val = 0.0  # Use 0.0 to indicate exp_decay
    else:
        res_iter_v_threshold_val = float(res_iter_v_threshold)
    
    return torch.tensor([
        int(params.get('mesh_x', 64)),
        int(params.get('mesh_y', 16)), 
        params.get('omega_u', 0.6),
        params.get('omega_v', 0.6),
        params.get('omega_p', 0.3),
        params.get('diff_u_threshold', 1e-07),
        params.get('diff_v_threshold', 1e-07),
        res_iter_v_threshold_val
    ], dtype=torch.float32)

def ns_channel_2d_preprocessor(tunable_tensor, result_tensor, preproc_success=False, preproc_cost=False):
    """Preprocess result tensor for ns_channel_2d"""
    processed = result_tensor.clone()
    if preproc_cost:
        mesh_x = tunable_tensor[0]
        mesh_y = tunable_tensor[1]
        # Normalize cost by mesh size
        processed[1] = result_tensor[1] / (mesh_x * mesh_y)
    return processed

def ns_transient_2d_static_mapper(profile_path, level):
    """Map static parameters for ns_transient_2d to tensor format"""
    import torch
    static_params = extract_static(profile_path)
    
    # One-hot encoding for boundary condition (1-6)
    boundary_condition = static_params.get('boundary_condition', 1)
    boundary_onehot = torch.zeros(6)
    if 1 <= boundary_condition <= 6:
        boundary_onehot[boundary_condition - 1] = 1  # Convert to 0-indexed
    
    # Reynolds number encoding (1k=0, 100k=1)
    reynolds_num = static_params.get('reynolds_num', 1000.0)
    reynolds_encoding = 1 if reynolds_num >= 50000 else 0  # 100k vs 1k
    
    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level, 1)  # Default to medium
    
    # Create tensor with reynolds and level encoding
    static_tensor = torch.tensor([
        reynolds_encoding,
        level_encoding
    ], dtype=torch.float32)
    
    return torch.concatenate([boundary_onehot, static_tensor], dim=-1)

def ns_transient_2d_tunable_mapper(params):
    """Map tunable parameters for ns_transient_2d to tensor format"""
    return torch.tensor([
        params.get('cfl', 0.05),
        int(params.get('resolution', 200)),
        params.get('relaxation_factor', 1.3),
        params.get('residual_threshold', 1e-02)
    ], dtype=torch.float32)

def epoch_static_mapper(profile_path, level):
    """Map static parameters for epoch_nx to tensor format"""
    import torch
    static_params = extract_static(profile_path)

    # Use level encoding: low=0, medium=1, high=2
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    level_encoding = level_map.get(level, 1)  # Default to medium

    return torch.tensor([
        static_params.get('a0', 200.0),        # Normalized laser amplitude
        static_params.get('n_target', 5.0),   # Density of target [n_cr]
        level_encoding
    ], dtype=torch.float32)

def epoch_tunable_mapper(params):
    """Map tunable parameters for epoch_nx to tensor format"""
    return torch.tensor([
        int(params.get('nx', 400)),              # Number of grid cells
        params.get('dt_multiplier', 0.95),       # Time-increment multiplier
        int(params.get('npart', 20)),           # Number of particles per cell
    ], dtype=torch.float32)

        