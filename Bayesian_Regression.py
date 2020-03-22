'''
Bayesian Regression - Introduction (Part 1)  http://pyro.ai/examples/bayesian_regression.html
https://www.jianshu.com/p/d6fd623b2450
'''

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # statistical data visualization (based on matplotlib.)
from functools import partial

import pyro
import pyro.distributions as dist


from torch import nn
from pyro.nn import PyroModule

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

# 测试的flag
smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)
pyro.set_rng_seed(1)


# rugged表示某国地形的崎岖程度；
# cont_africa表示某国是否在非洲；
# rgdppc_2000表示在2000年某国的人均GDP

DATA_URL = 'https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv'
data = pd.read_csv(DATA_URL, encoding='ISO-8859-1')
df = data[['rugged', 'cont_africa', 'rgdppc_2000']]
df = df[np.isfinite(df.rgdppc_2000)]
df['rgdppc_2000'] = np.log(df['rgdppc_2000'])   # 由于人均GDP差别较大，我们先将其取对数。



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df['cont_africa'] == 1]
non_african_nations = df[df['cont_africa'] == 0]


sns.scatterplot(non_african_nations['rugged'],
                non_african_nations['rgdppc_2000'],
                ax=ax[0])
ax[0].set(xlabel='Terrain Ruggedness Index',
          ylabel='log GDP (2000)',
          title='Non African Nations')

sns.scatterplot(african_nations['rugged'],
                african_nations['rgdppc_2000'],
                ax=ax[1])
ax[1].set(xlabel='Terrain Ruggedness Index',
          ylabel='log GDP (2000)',
          title='African Nations')
plt.show()


# 增加一个变量来捕捉'cont_africa'和'rugged'的关系
df['cont_africa_x_rugged'] = df['cont_africa'] * df['rugged']
data = torch.tensor(df[['cont_africa', 'rugged', 'cont_africa_x_rugged', 'rgdppc_2000']].values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]


'''
build linear regression
'''
linear_reg_model = PyroModule[nn.Linear](3, 1)

# 定义损失和优化器
loss_fn = torch.nn.MSELoss(reduction='sum')    # 使用均方差（MSE）损失作为优化的目标函数
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
num_iterations = 1500 if not smoke_test else 2

def train():
    # data前向传播
    y_pred = linear_reg_model(x_data).squeeze(-1)
    # 均方差损失
    loss = loss_fn(y_pred, y_data)
    # 将梯度置为零
    optim.zero_grad()
    # 梯度反传
    loss.backward()
    # 梯度更新一步
    optim.step()
    return loss

for j in range(num_iterations):
    loss = train()
    if (j + 1) % 50 == 0:
        print('[iteration %04d] loss: %.4f' %(j + 1, loss.item()))

# 监听参数的变化
print('Learned parameters:')
for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy())




# draw the regression results

fit = df.copy()    # copy（） 和 copy（deep= False）都是浅拷贝，是复制了旧对象的内容，然后重新生成一个新对象，新旧对象的ID 是不同的，改变旧对象不会影响新对象。
fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = fit[fit["cont_africa"] == 1]
non_african_nations = fit[fit["cont_africa"] == 0]
fig.suptitle("Regression Fit", fontsize=16)
ax[0].plot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], "o")
ax[0].plot(non_african_nations["rugged"], non_african_nations["mean"], linewidth=2)
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
ax[1].plot(african_nations["rugged"], african_nations["rgdppc_2000"], "o")
ax[1].plot(african_nations["rugged"], african_nations["mean"], linewidth=2)
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")
plt.show()



'''
 build Bayesian Regression
'''
from pyro.nn import PyroSample

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample('sigma', dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', dist.Normal(mean, sigma), obs=y)
        return mean


from pyro.infer.autoguide import AutoDiagonalNormal

# Instantiate the class
model = BayesianRegression(3, 1)
guide = AutoDiagonalNormal(model)


# opimizer
from pyro import optim
from pyro.infer import SVI, Trace_ELBO

adam = optim.Adam({'lr':0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())


# optimization process
pyro.clear_param_store()
for j in range(num_iterations):
    # 计算损失，梯度反传并更新
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print('[iteration %04d] loss: %.4f' % (j + 1, loss / len(data)))



# 通过查看Pyro的参数容器，我们检查优化后的参数
guide.requires_grad_(False)
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))


guide.quantiles([0.25, 0.5, 0.75])



from pyro.infer import Predictive


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


predictive = Predictive(model, guide=guide, num_samples=800,
                        return_sites=("linear.weight", "obs", "_RETURN"))
samples = predictive(x_data)
pred_summary = summary(samples)


mu = pred_summary["_RETURN"]
y = pred_summary["obs"]
predictions = pd.DataFrame({
    "cont_africa": x_data[:, 0],
    "rugged": x_data[:, 1],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_gdp": y_data,
})

# draw mu_perc_5
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = predictions[predictions["cont_africa"] == 1]
non_african_nations = predictions[predictions["cont_africa"] == 0]
african_nations = african_nations.sort_values(by=["rugged"])
non_african_nations = non_african_nations.sort_values(by=["rugged"])
fig.suptitle("Regression line 90% CI", fontsize=16)
ax[0].plot(non_african_nations["rugged"],
           non_african_nations["mu_mean"])
ax[0].fill_between(non_african_nations["rugged"],
                   non_african_nations["mu_perc_5"],
                   non_african_nations["mu_perc_95"],
                   alpha=0.5)
ax[0].plot(non_african_nations["rugged"],
           non_african_nations["true_gdp"],
           "o")
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
idx = np.argsort(african_nations["rugged"])
ax[1].plot(african_nations["rugged"],
           african_nations["mu_mean"])
ax[1].fill_between(african_nations["rugged"],
                   african_nations["mu_perc_5"],
                   african_nations["mu_perc_95"],
                   alpha=0.5)
ax[1].plot(african_nations["rugged"],
           african_nations["true_gdp"],
           "o")
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

plt.show()


# draw y_perc_95
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
ax[0].plot(non_african_nations["rugged"],
           non_african_nations["y_mean"])
ax[0].fill_between(non_african_nations["rugged"],
                   non_african_nations["y_perc_5"],
                   non_african_nations["y_perc_95"],
                   alpha=0.5)
ax[0].plot(non_african_nations["rugged"],
           non_african_nations["true_gdp"],
           "o")
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
idx = np.argsort(african_nations["rugged"])

ax[1].plot(african_nations["rugged"],
           african_nations["y_mean"])
ax[1].fill_between(african_nations["rugged"],
                   african_nations["y_perc_5"],
                   african_nations["y_perc_95"],
                   alpha=0.5)
ax[1].plot(african_nations["rugged"],
           african_nations["true_gdp"],
           "o")
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations")

plt.show()



weight = samples["linear.weight"]
weight = weight.reshape(weight.shape[0], 3)
gamma_within_africa = weight[:, 1] + weight[:, 2]
gamma_outside_africa = weight[:, 1]
fig = plt.figure(figsize=(10, 6))
sns.distplot(gamma_within_africa, kde_kws={"label": "African nations"},)
sns.distplot(gamma_outside_africa, kde_kws={"label": "Non-African nations"})
fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness")

plt.show()



