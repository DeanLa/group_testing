from scipy.stats import beta
import pymc as pm
from pymc import rbeta
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


class Sample(object):
    def __init__(self, q, n=1000000):
        self.q = q
        self.n = n
        self.G = int(np.ceil(q*n))
        self.B = n - self.G
        self.line = self.generate()
        
    def generate(self, from_file = None, seed=2016):
        self.seed = seed
        
        if from_file:
            name = "./samples/{}_{}.txt".format(str(self.q).replace(".",""),str(self.seed))
            self.line = np.loadtxt(name, dtype=np.int8)
            
        seed = np.random.seed(seed)
        line = np.append(np.ones(self.G, dtype = np.int8), np.zeros(self.B, dtype = np.int8))
        self.line = np.random.permutation(line)
        return self.line
    
    def save(self):
        name = "./samples/{}_{}.txt".format(str(self.q).replace(".",""),str(self.seed))
        np.savetxt(name, self.line,fmt="%d")


class Strategy(object):
    '''Gets a sample object and builds a test plan'''
    def __init__(self, sample, startig_test=1):
        self.q = beta(1,1)
        self.sample = sample
        self.queue = sample.line
        self.Ntests = 0
        self.Htests = 0
        self.test_now = startig_test
        self.G = 0
        self.B = 0
        self.q_history = []
        self.H_history = []
        self.name = ""
        self.gsize = 0
        self.failed_h = 0
    def test(self):
        '''test needs to be implemented according to the relevant strategy'''
        pass
        raise NotImplementedError()
        
        
    def g_test(self, items):
        if items.size == 1:
            return
        self.gsize += 1
        self.Ntests += 1
        m = int(items.size/2)
        test, leave = np.split(items, [m])
        if all(test):
            self.G += test.size
            self.g_test(leave)
        else:
            # Leave is now binomial and back to queue
            self.queue = np.append(leave, self.queue)
            self.g_test(test)
    
    def update(self, h_now):
        '''updates all histories'''
        self.Ntests += 1
        self.Htests += 1
        self.H_history.append(h_now)
        self.q_history.append(self.G / ( 1+ self.G + self.B))
        
    def plot(self, fig = None):
        l = len(self.H_history)
        if fig is None:
            fig, ax = plt.subplots(2,figsize=(16,12))
        ax = fig.get_axes()
        
        # Axis 0 - q History
        ax[0].set_title("q History")
        ax[0].plot(self.q_history, label=self.name)
        ax[0].set_xlim(0,l)
        ax[0].legend(loc='best')
        
        # Axis 1 - H history
        ax[1].set_title("H History")
        ax[1].plot(self.H_history[::1], label = self.name)
#         ax[1].plot([0,l],[np.mean(self.H_history)]*2, ls='--', lw=2, c='b',
#                    label="{} Mean = {:.4f}".format(self.name, np.mean(self.H_history)))
        ax[1].set_xlim(0,l)
        ax[1].legend()
        
        
        return fig, ax
    
# Double or Divide Strategy
class DOD(Strategy):
    def test(self):
        self.name = "Double or Divide"
        h_now = self.test_now
        while self.queue.size > 0:
            # Update histories
            self.update(h_now)
            items, self.queue = np.split(self.queue,[h_now])
            if all(items):
                self.G += items.size
                h_now *= 2
            else:
                self.B +=1
                self.g_test(items)
                h_now = max(1, h_now//2)

class Xwise(Strategy):
    '''Implements a constant test of size :starting_test:'''
    def test(self):
        h_now = self.test_now
        self.name = "{}-wise".format(h_now)
        while self.queue.size > 0:
            self.update(h_now)
            items, self.queue = np.split(self.queue,[h_now])
            if all(items):
                self.G += items.size
            else:
                self.B += 1
                self.g_test(items) 
#                 print (self.gsize, end=" ")
                self.failed_h += 1
        self.gsize /= self.failed_h
                
class OneByOne(Xwise):
    def test(self):
        self.test_now = 1
        super().test()
        self.name = "One by One"


                
class Pairwise(Xwise):
    def test(self):
        self.test_now = 2
        super().test()
        self.name = "Pairwise"
                
class NonAdaptiveDecision(Strategy):
    def test(self):
        self.name = "Non Adaptive"
        h_now = self.test_now
        after = 1000
        while self.queue.size > 0:
            # Update histories
            self.update(h_now)
            last_q = self.q_history[-1]
            if (self.G + self.B > after) and (last_q < 0.835):
                break
            items, self.queue = np.split(self.queue,[h_now])
            if all(items):
                self.G += items.size
                h_now *= 2
            else:
                self.B +=1
                self.g_test(items)
                h_now = max(1, h_now//2)
                
        if last_q < 0.618:
            # Do one by one
            self.name += " -> 1by1"
            while self.queue.size > 0:
                self.update(1)
                items, self.queue = np.split(self.queue,[1])
                if all(items):
                    self.G += items.size
                else:
                    self.B +=1
        elif last_q < 0.835:
            # Do Pairwise
            self.name += " -> pairwise"
            while self.queue.size > 0:
                self.update(2)
                items, self.queue = np.split(self.queue,[2])
                if all(items):
                    self.G += items.size
                else:
                    self.B += 1
                    self.g_test(items)
        else:
            self.name += " -> DOD"
            
class Langsam(Strategy):
    '''Langsam Adaptive Bayesian strategy.'''
    def test(self, function_g = lambda x: x, function_b = lambda x: 1,seed = 2):
        '''function_g and function_b are the functions on number of goods or number of bads to add to
        the beta draw. Default is H-test if GOOD, or 1 if BAD'''
#         np.random.seed(seed)
#         pm.numpy.random.seed(seed)
        self.name = "Langsam Adaptive Bayesian"
        l = 10
        powers = np.arange(0,l,1)
#         sizes = 2**powers
        sizes = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30])
        l = len(sizes)
        Gs = np.zeros(l)
        Bs = np.zeros(l)
        choice_history = []
        
        
        f = open('./test.txt', 'w')
        f2 = open('./draws.txt', 'w')
        while self.queue.size > 0:
            draw = rbeta(1 + np.log2(sizes) + Gs, 1 + Bs)
#             f2.write("{}\n\n".format(draw))
            draw = np.argmax(draw)
            h_now = sizes[draw]
            choice_history.append(h_now)
            
            f.write ("{} ".format(h_now))
            # Update histories
            self.update(h_now)
            q_now = self.q_history[-1]
            items, self.queue = np.split(self.queue,[h_now])
            if all(items):
                self.G += items.size
#                 Gs[draw] += function_g(items.size)
                Gs[draw] += (draw+1)/10
                f.write ("SUCCESS ")
                # Update
            else:
                f.write ("_______ ")
                self.B += 1
                self.g_test(items)
#                 Bs[draw] += function_b(items.size)
                Bs[draw] += 1 / (1 + draw)
            f.write ("({:.3f},{:.3f})\n".format((np.log2(sizes) + 1 + Gs)[draw], 1 + Bs[draw]))
        f.close()
        f2.close
#         print(sizes,'\n',1 + Gs,'\n', 1 + Bs)
#         print (np.histogram(choice_history, bins=sizes)[0])
# #         print (choice_history)
#         print()
        self.choices = choice_history
    
    def hist(self):
        fig, ax = plt.subplots()
        ax.hist(self.choices, bins=2**np.arange(0,10,1))


# Functions

def min_change(arr):
    min_arr = arr.argmin(axis=1)
    roll = np.roll(min_arr,-1)
    change = roll - min_arr != 0
    return change
    