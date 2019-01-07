from __future__ import print_function
import numpy as np
import scipy.stats as st
import multiprocessing as mp
from collections.abc import Iterable

np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights = arg
    return get_reward_func(weights)

class WeightUpdateStrategy:
    __slots__ = ("learning_rate",)
    def __init__(self, dim, learning_rate):
        self.learning_rate = learning_rate


class strategies:
    class GD(WeightUpdateStrategy):
        def update(self, i, g):
            return self.learning_rate * g


    class Adam(WeightUpdateStrategy):
        __slots__ = ("eps", "beta1", "beta2", "m", "v")
        def __init__(self, dim, learning_rate, eps=1e-8, beta1=0.9, beta2=0.999):
            super().__init__(dim, learning_rate)
            self.eps = eps
            self.beta1 = beta1
            self.beta2 = beta2
            self.m = np.zeros(dim)
            self.v = np.zeros(dim)

        def update(self, i, g):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (g**2)
            return self.learning_rate * np.sqrt(1-self.beta2) / (1-self.beta1) * self.m[i] / np.sqrt(np.sqrt(self.v[i])+self.eps)


class EvolutionStrategy(object):
    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.999,
                 num_threads=1, limits=None, printer=None, distributions=None, strategy=None):
        if limits is None:
            limits = (np.inf, -np.inf)
        self.weights = weights
        self.limits = limits
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        if distributions is None:
            distributions = st.norm(loc=0., scale=sigma)
        if isinstance(distributions, Iterable):
            distributions = list(distributions)
            self.SIGMA = np.array([d.std() for d in distributions])
        else:
            self.SIGMA = distributions.std()

        self.distributions = distributions
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        if printer is None:
            printer = print
        self.printer = printer
        if strategy is None:
            strategy = strategies.GD
        self.strategy = strategy(len(weights), self.learning_rate)

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            weights_try.append(w[index] + i)
        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            if isinstance(self.distributions, Iterable):
                for j, w in enumerate(self.weights):
                    x.append(self.distributions[j].rvs(*w.shape))
            else:
                for w in self.weights:
                    x.append(self.distributions.rvs(*w.shape))

            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)
            rewards = pool.map(worker_process, worker_args)

        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        return rewards

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        grad_factor = 1. / (self.POPULATION_SIZE * (self.SIGMA ** 2))

        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            corr = np.dot(layer_population.T, rewards).T

            if not isinstance(grad_factor, np.ndarray):
                g = grad_factor * corr
            else:
                g = grad_factor[index] * corr
            self.weights[index] = w + self.strategy.update(index, g)
        self.learning_rate *= self.decay

    def run(self, iterations, print_step=10):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for iteration in range(iterations):

            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)

            if (iteration + 1) % print_step == 0:
                self.printer('iter %d. reward: %f' % (iteration + 1, self.get_reward(self.weights)), (self.weights if self.weights.shape[0] <= 10 else None) )
        if pool is not None:
            pool.close()
            pool.join()
