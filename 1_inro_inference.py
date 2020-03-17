import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)


def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))


def conditioned_scale():
    condi_scale = pyro.condition(scale, data={"measurement": 9.5})
    return condi_scale


def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))





if __name__ == "__main__":

    guess = 8.5

    pyro.clear_param_store()
    svi = pyro.infer.SVI(model=conditioned_scale(),
                         guide=scale_parametrized_guide,
                         optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                         loss=pyro.infer.Trace_ELBO())


    losses, a,b  = [], [], []
    num_steps = 2500
    for t in range(num_steps):
        losses.append(svi.step(guess))
        a.append(pyro.param("a").item())
        b.append(pyro.param("b").item())

    plt.figure(1)
    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")


    print('a = ',pyro.param("a").item())
    print('b = ', pyro.param("b").item())



    plt.subplot(1,2,1)
    plt.plot([0,num_steps],[9.14,9.14], 'k:')
    plt.plot(a)
    plt.ylabel('mu')

    plt.subplot(1,2,2)
    plt.ylabel('sigma')
    plt.plot([0,num_steps],[0.6,0.6], 'k:')
    plt.plot(b)
    plt.tight_layout()



    plt.figure(2)
    from scipy.stats import norm

    mu = pyro.param("a").item()
    sigma = pyro.param("b").item()

    x_axis = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
    plt.title("weight: mu = 9.11 sigma = 0.6")
    plt.show()
