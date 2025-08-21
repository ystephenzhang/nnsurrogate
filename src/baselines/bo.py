from bayes_opt import BayesianOptimization
import hydra
import functools
import datetime

import sys
sys.path.append("/home/ubuntu/dev/src")
from general_utils import Verifier

def heat_1d_optimize(profile, task, cfl, n_space):
    ver = Verifier("1D_heat_transfer", task, tolerance="high")
    _, _, score = ver.metric({
        "cfl": cfl,
        "n_space": int(n_space)
    }, profile)
    return score

# Bounded region of parameter space
#pbounds = {'cfl': (0, 1), 'n_space': (64, 2048)}
pbounds = {'n_space':(64, 2048)}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    problem = cfg.problem
    profile = cfg.profile
    task = cfg.task

    save_dir = cfg.save_dir
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )
    save_path = f"{save_dir}/bo-{problem}-{profile}-{task}-{datetime_str}.txt"
    
    if problem == "1D_heat_transfer":
        if task == "cfl":
            black_box_function = functools.partial(
                                                heat_1d_optimize, 
                                                profile=profile,
                                                task=task,
                                                n_space=100)
        if task == "n_space":
            black_box_function = functools.partial(
                                            heat_1d_optimize,
                                            profile=profile, 
                                            task=task,
                                            cfl=0.25)
    else:
        raise NotImplementedError
    
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(init_points=cfg.init_points, n_iter=cfg.n_iter)
    
    with open(save_path, 'w') as f:
        for i, res in enumerate(optimizer.res):
            f.write("Iteration {}: \n\t{}\n".format(i, res))
        max_log = str(optimizer.max)
        f.write(f"\n{max_log}")
        
if __name__ == "__main__":
    main()