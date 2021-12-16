import random

import numpy as np
from scipy.stats import norm, truncnorm
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot(l,e):
    l_avg=np.mean(l,axis=0)
    e_avg=np.mean(e,axis=0)
    print(e_avg[-1])
    print(l_avg[-1])
    samples=range(0,1000)
    plt.plot(samples, l_avg, label='lambda algorithm ',color='g')
    plt.plot(samples, e_avg, label='eta algorithm ',color='b')
    plt.title("online decision making algorithms performance", fontsize=12)
    plt.xlabel("samples", fontsize=12)
    plt.ylabel("expected performance ratio", fontsize=12)
    plt.grid(True)
    plt.ylim([0, 1])
    plt.legend()
    plt.show()



def create_dists():
    dists = []
    for _ in range(dist_number):
        mean = np.random.rand() * mean_range
        std = np.random.rand() * std_range
        dist = norm(mean, std)
        interval = sorted([dist.rvs(), dist.rvs()])
        truncated_dist = truncnorm(interval[0], interval[1], loc=mean, scale=std)
        dists.append(truncated_dist)
    return dists

def get_x(dists):
    x=[]
    for i in dists:
        x.append(i.rvs())
    return x

def lambda_th(x,dists):
    x_max=np.argmax(x)
    landa=0.5*dists[x_max].mean()
    for i in x:
        if i>=landa:
            return i
    return x[-1]

def eta_th(x,dists):
    x_max = np.argmax(x)
    eta=dists[x_max].ppf(0.5)
    for i in x:
        if i>=eta:
            return i
    return x[-1]

def prophet(x):
    return np.max(x)

def k_1_search():
    total_lambda=[]
    total_eta=[]
    for s in range(seed_number):
        random.seed(seed_number)
        np.random.seed(seed_number)
        dists = create_dists()
        lambda_results=[]
        eta_results=[]
        prophet_results=[]
        expected_lambda_performance=[]
        expected_eta_performance=[]
        for _ in tqdm(range(samples_number)):
            x=get_x(dists)
            prophet_results.append(prophet(x))
            eta_results.append(eta_th(x,dists))
            lambda_results.append(lambda_th(x,dists))
            expected_eta_performance.append(np.mean(eta_results)/np.mean(prophet_results))
            expected_lambda_performance.append(np.mean(lambda_results)/np.mean(prophet_results))
        total_eta.append(expected_eta_performance)
        total_lambda.append(expected_lambda_performance)

    return total_lambda,total_eta


if __name__ == "__main__":
    dist_number=100
    mean_range=10
    std_range=1
    samples_number=1000
    seed_number=5
    total_lambda,total_eta=k_1_search()
    plot(total_lambda,total_eta)