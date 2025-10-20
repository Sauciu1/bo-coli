import pickle
from src.gp_and_acq_f.custom_gp_and_acq_f import MaternKernelSGP
from src.ax_helper import SequentialRuns, BatchClientHandler
import multiprocessing as mp
import time
import numpy as np
import copy

with open("data/mevalonate_pathway/signal_pipeline.pkl", "rb") as f:
    signal_pipeline = pickle.load(f)
with open("data/mevalonate_pathway/noise_pipeline.pkl", "rb") as f:
    noise_pipeline = pickle.load(f)
with open("data/mevalonate_pathway/parameters.pkl", "rb") as f:
    parameters = pickle.load(f)


def test_function(**kwargs):
    """Passes parameters to the signal model and returns the predicted response."""
    X = np.array([[kwargs[p] for p in parameters]])
    resp = signal_pipeline.predict(X)[0]
    return resp


def noise_function(x, y):
    """Adds noise to a clean response based on the noise model."""
    X = np.array([[x[p] for p in parameters]])
    noise_std = max(0.001, noise_pipeline.predict(X)[0])
    noise = float(y + np.random.normal(0, noise_std))
    return noise



with open("data/mevalonate_pathway/empirical_tester.pkl", "rb") as f:
    tester: SequentialRuns = pickle.load(f)



def _single_run():
    """this will be rerun by mcp"""


    #### there 568 unique observations, we treat it as our budget
    runs = copy.copy(tester).run(
        MaternKernelSGP,
        n_runs=int(568/6/6),
        technical_repeats=6,
        batch_size=6,
        noise_fn=noise_function,
        plot_each=True,
    )

    obs = runs.get_grouped_obs()
    return obs


print(__name__)
mode = "multicore"


##
def run_grid(save_path, param_grid):

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
            job = pool.apply_async(_single_run, args=[param])
            jobs.append((job, key, param))

        for job, param_key, original_param in jobs:
            result = job.get()
            # Optionally store original param dict alongside result
            save_result(result, param_key)

    t1 = time.perf_counter()
    print(f"All tasks completed in {t1 - t0:.2f} seconds.")


param_grid = [
    {
        "rerun": rerun,
    }
    for rerun in range(50)
]


if __name__ == "__main__":
    save_dir = "data/mevalonate_pathway/sim/"
    time_str = time.strftime("%d_%m_%Y_%H_%M")
    save_PATH = save_dir + f"Matern_modelling_{time_str}.pkl"
    print(f"Saving to {save_PATH}")
    run_grid(save_path=save_PATH, param_grid=param_grid)
