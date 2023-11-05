import numpy as np 
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.optimize as optim
'''
class Pareto(st.rv_continuous):
    

    def __init__(self, gamma, shift, xmin):
        super(Pareto, self).__init__(a=shift)
        self.gamma = gamma
        self.shift = shift
        self.xmin = xmin

    def _pdf(self, x):
        normalizer = (self.gamma - 1) * (self.xmin - self.shift) ** (self.gamma - 1)
       # print(x - self.shift)
        if (((x - self.shift) ** self.gamma ) > 1000000000):
            return 0
        else:
            return normalizer / ((x - self.shift) ** self.gamma )
'''

class Simulate:


    def __init__(self):
        self.gamma = 1.1

    def simulate(self, num_samples):
        naive_samps = st.pareto(b=self.gamma).rvs(size=1000000) - 1
        return naive_samps


class MLE:

    def __init__(self):
        pass 

    def maxim(self):
        pass


    def loss(self):
        pass 


    
if __name__ == "__main__":
    simulate = Simulate()
    samps = simulate.simulate(1000)
    print(st.pareto.fit(samps))
