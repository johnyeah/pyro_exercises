import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch

import pyro
import pyro.infer
import pyro.optim
from pyro.optim import Adam
import torch.distributions.constraints as constraints
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO


pyro.set_rng_seed(101)


# 清空参数的容器
pyro.clear_param_store()

# 创建观察数据。这里假定实验结果为，前6次为正，后4次为反
data = list()
for _ in range(6):
    data.append(torch.tensor(1.))

for _ in range(4):
    data.append(torch.tensor(0.))


def model(data):
    # 先验的 beta 分布的超参
    alpha0 = torch.tensor(10.)
    beta0 = torch.tensor(10.)
    # 从先验分布中采样f
    f = pyro.sample('latent_fairness', dist.Beta(alpha0, beta0))
    # 遍历整个观察数据集
    for i in range(len(data)):
        # 似然函数在数据点i服从伯努利分布
        pyro.sample('obs_{}'.format(i), dist.Bernoulli(f), obs=data[i])


def guide(data):
    # 在Pyro中注册变分分布的参数
    # 两个参数值均为15.0
    # 我们对没有约束的参数采用梯度下降
    # 注意，这里是pyro.param，不是pyro.sample!!!
    alpha_q = pyro.param('alpha_q', torch.tensor(15.), constraint=constraints.positive)
    beta_q = pyro.param('beta_q', torch.tensor(15.), constraint=constraints.positive)
    # 从Beta(alpha_q, beta_q)采样得到latent_fairness
    pyro.sample('latent_fairness', dist.Beta(alpha_q, beta_q))

# 设置优化器参数
adam_params = {'lr': 0.0005, 'betas': (0.9, 0.999)}
optimizer = Adam(adam_params)

# 设置推断算法
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())


n_steps = 500


# 梯度下降
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end=' ')

# 监听变分参数的值
alpha_q = pyro.param('alpha_q').item()
beta_q = pyro.param('beta_q').item()

# 根据beta分布的特点， 我们计算推断出的公平性系数
inferred_mean = alpha_q / (alpha_q + beta_q)
# 计算其标准差
factor = beta_q / (alpha_q * (1. + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print('\nbased on the data and our prior belief, the fairness '+
        'of the coin is %.3f +- %.3f' % (inferred_mean, inferred_std))