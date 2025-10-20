
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd



def plot_gp(gp:GaussianProcessRegressor, X:np.array, y:np.array, X_train:np.array, y_train:np.array):
    gp.fit(X_train, y_train)
    mean_prediction, std_prediction = gp.predict(X, return_std=True)


    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
 
    )

    sns.scatterplot(x=X_train.ravel(), y=y_train, marker='x', color='r', label='Noisy Observations', s=100)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted", color='k')

    df = pd.DataFrame({'X_train': X_train.ravel(), 'y_train': y_train.ravel()})
    grouped = df.groupby('X_train')['y_train'].mean()
    X_unique = grouped.index.values
    y_matched = grouped.values



    std = df.groupby('X_train')['y_train'].std().values
    noise_alpha = 1.96 * std


    plt.errorbar(
        X_unique,
        y_matched,
        noise_alpha,
        linestyle="None",
        color="k",
        marker = '.',
        markersize=10,
        label="Observations",
    )

    plt.legend(loc="upper left")
    plt.xlabel("Input $x$")
    plt.ylabel("Output $y$")
    plt.title("Gaussian Process Regression")
    fig = plt.gcf()
    return fig






def sincx(x):
    return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)

def linear_func(x, slope=1.0, intercept=0.0):
    return slope * x + intercept

def poly_func(roots, coeffs, x):
    p = torch.poly1d(coeffs)
    return p(x)


if __name__ == "__main__":
    x = torch.linspace(-10, 10, 1000)
    y_sincx = sincx(x)
    y_linear = linear_func(x, slope=2.0, intercept=1.0)
    y_poly = poly_func(roots=[-3, 1, 2], coeffs=[1, -0.5, -4, 6], x=x)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x.numpy(), y_sincx.numpy())
    plt.title("Sinc Function")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(x.numpy(), y_linear.numpy())
    plt.title("Linear Function")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(x.numpy(), y_poly.numpy())
    plt.title("Polynomial Function")
    plt.grid()

    plt.tight_layout()
    plt.show()
