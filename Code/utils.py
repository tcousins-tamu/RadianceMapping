
#utils for the radiance mapping 
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
#Weights class created to speed up the gsolve function
class Weights():
    def __init__(self, Zmax=None, Zmin=None):
        self.Zmax = Zmax
        self.Zmin = Zmin
        self.Zavg = .5*(self.Zmax+self.Zmin)
        
    def __getitem__(self, z):
        if isinstance(z, Iterable):
            cpy = deepcopy(z)
            leq = np.where(z<=self.Zavg)
            g = np.where(z>self.Zavg)
            cpy[leq] = cpy[leq] - self.Zmin
            cpy[g] =  -1*cpy[g] + self.Zmax
            return cpy
        
        if z<=self.Zavg:
            return z-self.Zmin
        else:
            return self.Zmax-z