import math
import random
import sys

import numpy as np
from scipy.stats import norm, truncnorm
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot(l):
    l_avg = np.mean(l, axis=0)
    print("average: "+str(l_avg[-1]))
    samples = range(0, 1000)
    plt.plot(samples, l_avg, color='g')
    plt.title("online decision making algorithms performance", fontsize=12)
    plt.xlabel("samples", fontsize=12)
    plt.ylabel("expected performance ratio", fontsize=12)
    plt.grid(True)
    plt.ylim([0, 1.2])
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
    x = []
    for i in dists:
        x.append(i.rvs())
    return x


def prophet(x, N):
    final_list = []
    main_list = x.copy()
    for i in range(0, N):
        max1 = 0

        for j in range(len(main_list)):
            if main_list[j] > max1:
                max1 = main_list[j]

        main_list.remove(max1)
        final_list.append(max1)
    return sum(final_list)


def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact


def threshold(k, dists):
    delta = 1 - (math.exp(-k) * math.pow(k, k)) / factorial(k)
    print("gama: " + str(delta))
    thresh = dists[0].mean()
    for i in dists:
        if i.mean() < thresh:
            thresh = i.mean()
    while True:
        probs = sum([1 - i.cdf(thresh) for i in dists]) / k
        if probs == delta or probs-delta<0.01:
            return thresh
        else:
            thresh = thresh + decay_rate


def algo(x, k, thresh):
    final_list = []
    for i in range(len(x)):
        if x[i] >= thresh:
            final_list.append(x[i])
        if len(x) - (i+1) == k-len(final_list):
            final_list.extend(x[i+1:])
        if len(final_list) == k:
            break
    if len(final_list)>12:
        sys.exit()
    return sum(final_list)


def k_general_search():
    total_performance = []
    for s in range(seed_number):
        random.seed(s)
        np.random.seed(s)
        dists = create_dists()
        results = []
        prophet_results = []
        expected_performance = []
        thresh = threshold(k, dists)
        for _ in tqdm(range(samples_number)):
            x = get_x(dists)
            prophet_results.append(prophet(x, k))
            results.append(algo(x, k, thresh))
            expected_performance.append(np.mean(results) / np.mean(prophet_results))
        total_performance.append(expected_performance)
    return total_performance


if __name__ == "__main__":
    decay_rate = 0.01
    dist_number = 64
    k = 6
    mean_range = 10
    std_range = 1
    samples_number = 1000
    seed_number = 5
    total_performance = k_general_search()
    plot(total_performance)