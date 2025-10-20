from src import ax_helper
from src.ax_helper import SequentialRuns
from src.toy_functions import Hartmann6D
import numpy as np
from ax import Client, RangeParameterConfig
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
import pickle
import time
import multiprocessing as mp
import os
from src.gp_and_acq_f.custom_gp_and_acq_f import HeteroWhiteSGP, GammaNoiseSGP

save_dir = "data/bayes_sim/"
param_range = [
    RangeParameterConfig(name=f"x{i+1}", parameter_type="float", bounds=(0.0, 1.0))
    for i in range(6)
]

metric_name = "response"
dim_names = [rp.name for rp in param_range]


def _single_run(params, gp_used):

    def noise_fn(x, y):
        return y + np.random.normal(0, params["noise"])

    local_tester = SequentialRuns(
        Hartmann6D().eval_at, param_range, dim_names, metric_name
    )

    runs = local_tester.run(
        gp_used,
        n_runs=params["cycles"],
        technical_repeats=params["technical_repeats"],
        batch_size=params["batches"],
        noise_fn=noise_fn,
        plot_each=False,
    )
    df_local = runs.get_batch_observations()

    return df_local


print(__name__)
mode = "multicore"


def run_grid(save_path, param_grid, gp_used):


    t0 = time.perf_counter()
    print("Starting batch Bayesian optimization tests...")

    n_workers = min(mp.cpu_count() - 4, 80)
    print(f"Detected {mp.cpu_count()} CPU cores. Using {n_workers} workers.")

    r_n_dict = {}

    def save_result(result, param_key):
        r_n_dict[param_key] = result
        with open(save_path, "wb") as f:
            pickle.dump(r_n_dict, f)
        print(
            f"Saved result for {param_key} - Total completed: {len(r_n_dict)}/{len(param_grid)}"
        )

    mp.set_start_method("spawn", force=True)  # Windows-safe

    def dict_key(d: dict) -> str:
        # Create a stable, hashable, human-readable key
        return "|".join(f"{key}={val}" for key, val in d.items())

    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        jobs = []
        for param in param_grid:
            key = dict_key(param)
            job = pool.apply_async(_single_run, args=[param, gp_used])
            jobs.append((job, key, param))

        for job, param_key, original_param in jobs:
            result = job.get()
            # Optionally store original param dict alongside result
            save_result(result, param_key)

    t1 = time.perf_counter()
    print(f"All tasks completed in {t1 - t0:.2f} seconds.")




local_test_param_grid = [
    {
        "technical_repeats": [1],
        "noise": noise,
        "cycles": [20],
        "batches": [1],
        "rerun": rerun,
    }
    for noise in [3.4 * x for x in [0.0, 0.1]]
    for rerun in range(3)
]
if __name__ == "__main__":

    gp_used = SingleTaskGP

    

    param_grid = [
        {
            "technical_repeats": technical_repeats,
            "noise": noise,
            "cycles": cycles,
            "batches": batch,
            "rerun": rerun,
        }
        for technical_repeats in [1, 2, 4, 8]
        for noise in [3.4 * x for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
        for batch in [1]
        for cycles in [60]
        for rerun in range(10)
    ]
    #param_grid = local_test_param_grid
    save_PATH = save_dir + f"run1_{gp_used.__name__}_broad_09_10_2025.pkl"
    print(f"Saving to {save_PATH}")
    run_grid(save_path=save_PATH, param_grid=param_grid, gp_used=gp_used)
