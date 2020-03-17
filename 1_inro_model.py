import torch
import pyro


# Normal distribution N(0,1)

loc = 0.   # mean zero
scale = 1. # unit variance
normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
x = normal.rsample() # draw a sample from N(0,1)
print("sample", x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)


# simple example
def weather():
  # first Bernoulli distr: cloudy
    cloudy = torch.distributions.Bernoulli(0.3).sample()   # define a distribution
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'   # convert to a string so that return values of weather

  # second Normal dist: temp
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()

    return cloudy, temp.item()




def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream



def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)



if __name__ == "__main__":
    for _ in range(3):
        print('weather', weather())

    print("geometric", geometric(0.5))