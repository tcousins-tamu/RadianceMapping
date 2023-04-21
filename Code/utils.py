
#utils for the radiance mapping 

#Weights class created to speed up the gsolve function
class Weights():
    def __init__(self, Zmax=None, Zmin=None):
        self.Zmax = Zmax
        self.Zmin = Zmin
        self.Zavg = .5*(self.Zmax+self.Zmin)
        
    def __getitem__(self, z):
        if z<=self.Zavg:
            return z-self.Zmin
        else:
            return self.Zmax-z