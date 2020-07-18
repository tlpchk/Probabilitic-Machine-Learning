"""Utility functions for MCMC."""
import pyro


def run_mcmc(kernel, model_args, num_samples, warmup_steps):
    """Executes MCMC using given `kernel` and `data`."""
    pyro.clear_param_store()

    mcmc_posterior = pyro.infer.MCMC(
        kernel=kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
    )

    mcmc_posterior.run(*model_args)
    return mcmc_posterior


def sample_mcmc(mcmc_posterior):
    """Extracts samples from MCMC posterior."""
    return {
        k: v.detach().cpu()
        for k, v in mcmc_posterior.get_samples().items()
    }
