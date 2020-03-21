import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)

def model(data):
    # 定义beta分布的超参
    alpha0 = torch.tensor(10.)
    beta0 = torch.tensor(10.)
    # 从先验分布采样
    f = pyro.sample('latent_fairness', dist.Beta(alpha0, beta0))
    # 循环所有的观察数据
    for i in range(len(data)):
        # 观察的数据点 i 服从伯努利分布
        # 似然为 Bernoulli(f)
        pyro.sample('obs_{}'.format(i), dist.Bernoulli(f), obs=data[i])


def guide(data):
    # 注册两个变分参数
    alpha_q = pyro.param('alpha_q', torch.tensor(15.), constraint = constraints.positive)
    beta_q  = pyro.param('alpha_q', torch.tensor(15.), constraint = constraints.positive)
    # 采样Beta(alpha_q, beta_q)得到latent_fairness
    pyro.sample('latent_fairness', dist.Beta(alpha_q, beta_q))