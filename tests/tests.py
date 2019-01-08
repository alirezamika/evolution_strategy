#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
thisDir=Path(__file__).parent.absolute()
sys.path.append(str(thisDir.parent))

import numpy as np
import scipy.stats as st
from evostra import EvolutionStrategy

def modRosenbrockNP(X, a=1, b=100):
    return np.sqrt(np.power(a-X[0], 4) + b*np.power(X[1]-np.power(X[0], 2), 2))

def ackleyRosenbrockNp(X, a=20, b=0.2, c=2*np.pi):
    return np.real(a*(1-np.exp(-b*np.sqrt(modRosenbrockNP(X, a=0, b=a)/X.shape[0])))-np.exp(np.sum(np.cos(c*X), axis=0)/X.shape[0])+np.exp(1))


bounds = np.array([[0, 10], [-10, 10]])
initialPoint = np.array([10., 5.])

def get_reward(weights):
    weights=np.array(weights)
    #print(weights)
    res = -ackleyRosenbrockNp(weights)
    #print(res)
    return res


class OptimizersTests(unittest.TestCase):
    def testOptimizerSimple(self):
        es = EvolutionStrategy(initialPoint, get_reward, population_size=50, sigma=0.5, learning_rate=0.1, decay=1., num_threads=1)
        es.run(270, print_step=10)

    @unittest.skip
    def testOptimizerDistributions(self):
        es = EvolutionStrategy(initialPoint, get_reward, population_size=20, learning_rate=0.03, decay=1., num_threads=1, distributions=[st.norm(loc=0., scale=0.1), st.norm(loc=0., scale=0.2)])
        es.run(1000, print_step=1)


if __name__ == '__main__':
    unittest.main()
