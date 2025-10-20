from src.NNarySearch import NNarySearch
import pandas as pd
import torch

def logistic_tensor(x_0: float, k: float, start: float, end: float, steps: int = 100000) -> (torch.Tensor, torch.Tensor):
    """Generates a logistic function tensor.
    x_0: The x-value of the sigmoid's midpoint.
    k: The logistic growth rate.
    start: The start of the range.
    end: The end of the range.
    steps: The number of steps in the range.
    """
    linspace = torch.linspace(start, end, steps)

    logistic = 1 / (1 + torch.exp(-k *(linspace - x_0) ))
    return linspace, logistic



def run_simulations(center_space, power_space, split_space) -> pd.DataFrame:
    """Run simulations for different parameter combinations."""
    results = []
    for log_center in center_space:
        linspace, logistic = logistic_tensor(float(log_center), 1e-3, 0, 1e6)
        for splits in split_space:
            for power in power_space:
                power = round(power, 1)
                search= NNarySearch(splits, split_power=power)

                search.run_search(logistic)

                results.append({
                    "log_center": log_center,
                    "splits": splits,
                    "power": power,
                    "value": search.iterations,
                })
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index="splits",
        columns="power",
        values="value",
        aggfunc="mean"
    )

    return pivot





