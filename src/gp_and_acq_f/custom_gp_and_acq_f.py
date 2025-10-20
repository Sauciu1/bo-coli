# custom_gp_and_acq_f.py
"""This file contains custom Gaussian Process models and Acquisition Functions
For BOColi loader to discover your custom function it needs to have two attributes:
* `bocoli_name` - the name by which your class will be displayed in the user UI.
* `bocoli_type` - describes the type of class
    * `gp` indicates a gaussian process.
    * `acqf` indicates an acquisition function.
* `bocoli_description` is not mandatory, but makes tracking and choosing functions in the UI a bit simpler. 
If you are running from docker containers, you will need to rebuild the container
to see changes reflected in the UI.
"""
# Best of luck experimenting, best, Povilas


import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

dtype = torch.float64

def test_function():
    """Makes sure that the model does not load function instead of class"""
    pass
class bocoli_SingleTaskGP(SingleTaskGP):
    """A SingleTaskGP wrapper that works with botorch"""
    bocoli_name = "SingleTaskGP"
    bocoli_type = "gp"
    bocoli_description = """Standard Single Task Gaussian Process from BoTorch.
    Cannot handle noise, fits through every observed point"""

    def __init__(self, train_X, train_Y, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)

class GammaNoiseSGP(SingleTaskGP):
    """Just add a lot of assumed noise"""
    bocoli_name = "GammaNoiseSGP"
    bocoli_type = "gp"
    bocoli_description = """Single Task Gaussian Process with a Gamma prior on the noise."""
    def __init__(
        self,
        train_X,
        train_Y,
        noise_concentration: float = 1,
        noise_rate: float = 1,
    ):

        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(noise_concentration, noise_rate)
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
        )


class MaternKernelSGP(SingleTaskGP):
    """A SingleTaskGP with a Matern Kernel"""
    bocoli_name = "MaternKernelSGP"
    bocoli_type = "gp"
    bocoli_description = """Single Task Gaussian Process with a gamma noise priori Matern Kernel.
    No explicit technical repeat handling. """

    def __init__(self, train_X, train_Y, noise_concentration: float = 1.0, noise_rate: float = 1.0, **kwargs):
        matern_kernel = MaternKernel(nu=1.5)
        covar_module = ScaleKernel(matern_kernel)
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(noise_concentration, noise_rate)
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            **kwargs
        )


class HeteroNoiseSGP(SingleTaskGP):
    """Handles heteroscedastic noise by estimating the noise from repeated points."""
    def __init__(self, train_X, train_Y, **kwargs) -> None:
        std_unique, std_all, n_counts, inverse = self._calc_std(train_X, train_Y)

        if all(n_counts == 1):
            print("Warning: All points have only one repeat. defaulting to a Gamma Noise Kernel.")
            GammaNoiseSGP.__init__(self, train_X, train_Y, **kwargs)
        likelihood = FixedNoiseGaussianLikelihood(noise=std_all)


        
        super().__init__(train_X, train_Y, likelihood=likelihood, **kwargs)

    def _calc_std(self, train_X, train_Y):
        unique_X, inverse, n_counts = torch.unique(train_X, dim=0, return_inverse=True, return_counts=True)
        flat_y = train_Y.view(-1)
        std_list = []
        for i, x_val in enumerate(unique_X):
            mask = (inverse == i)
            y_std = flat_y[mask].std()

            std_list.append(y_std)
        std_list = torch.tensor(std_list, dtype=dtype)
        std_unique = std_list
        std_for_X = std_unique[inverse].to(train_Y.dtype).to(train_Y.device)
        return std_unique, std_for_X, n_counts, inverse
    

    

class HeteroWhiteSGP(HeteroNoiseSGP):
    """Heteroscedastic GP with a minimum (0.05 quantile) white noise level.
    Explicitly designed to handle noise from technical repeats."""
   # bocoli_name = "HeteroWhiteSGP"
    #bocoli_type = "gp"
   # bocoli_description = """Heteroscedastic Gaussian Process with a minimum white noise level.
   # Designed to handle noise from technical repeats. Defaults to Matern Kernel if no technical
   # repeats are present."""

    def __init__(self, train_X, train_Y, quintile=0.05, **kwargs) -> None:

        std_unique, std_for_X, n_counts, inverse = self._calc_std(train_X, train_Y)

        if all(n_counts == 1):
            print("Warning: All points have only one repeat. Consider using a different Kernel.")
            return MaternKernelSGP.__init__(self, train_X, train_Y, **kwargs)

        if torch.isnan((lower_noise := torch.quantile(std_unique, quintile))):
            lower_noise = std_unique.abs()[std_unique.abs() > 0].min()

        if torch.isnan((median_noise := torch.median(std_unique))):
            median_noise = std_unique.abs()[std_unique.abs() > 0].max()

        hetero_noise = std_for_X.clamp_min(lower_noise)

        # Set noise to median noise for points with only one repeat
        single_repeat_mask = n_counts == 1
        hetero_noise[torch.isin(inverse, torch.where(single_repeat_mask)[0])] = median_noise

        white_noise = torch.nn.Parameter(torch.tensor(lower_noise.item(), dtype=hetero_noise.dtype))



        uncertainty = hetero_noise + white_noise
        likelihood = FixedNoiseGaussianLikelihood(noise=uncertainty)


        return SingleTaskGP.__init__(self, train_X, train_Y, likelihood=likelihood, **kwargs)
    


from botorch.acquisition.knowledge_gradient import qKnowledgeGradient, qMultiFidelityKnowledgeGradient
from botorch.acquisition import qLogExpectedImprovement, qExpectedImprovement 

class bocoli_qLogExpectedImprovement(qLogExpectedImprovement):
    """A qLogExpectedImprovement"""
    bocoli_name = "qLogExpectedImprovement"
    bocoli_type = "acqf"

    def __init__(self, model, best_f=None, **kwargs):
        super().__init__(model, best_f=best_f, **kwargs)

class bocoli_qExpectedImprovement(qExpectedImprovement):
    """A qExpectedImprovement wrapper that works with botorch"""
    bocoli_name = "qExpectedImprovement"
    bocoli_type = "acqf"
    def __init__(self, model, best_f=None, **kwargs):
        super().__init__(model, best_f=best_f, **kwargs)

class bocoli_qKnowledgeGradient(qKnowledgeGradient):
    """A qKnowledgeGradient wrapper that works with botorch"""
    bocoli_name = "qKnowledgeGradient"
    bocoli_type = "acqf"
    def __init__(self, model, num_fantasies=64, **kwargs):
        super().__init__(model, num_fantasies=num_fantasies, **kwargs)

class bocoli_qMultiFidelityKnowledgeGradient(qMultiFidelityKnowledgeGradient):
    """A qMultiFidelityKnowledgeGradient wrapper that works with botorch"""
    bocoli_name = "qMultiFidelityKnowledgeGradient"
    bocoli_type = "acqf"
    def __init__(self, model, num_fantasies=64, **kwargs):
        super().__init__(model, num_fantasies=num_fantasies, **kwargs)




