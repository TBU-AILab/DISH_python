import random
import numpy as np
import math
from scipy.stats import cauchy

class Individual:
    def __init__(self, dim, bounds, id):
        self.features = [random.uniform(bounds[x][0], bounds[x][1]) for x in range(dim)]
        self.ofv = 0
        self.id = id

    def __repr__(self):
        return str(self.__dict__)

# simple test function
class Sphere:
    def __init__(self, dim):
        self.bounds = [[-100, 100] for x in range(dim)]

    def evaluate(self, x):
        sum = 0
        for i in range(len(x)):
            sum += x[i]**2
        return sum

###################
class DISH:
    
    def __init__(self, dim, maxFEs, OF, H, NP, minPopSize):
        self.dim = dim
        self.maxFEs = maxFEs
        self.NP = NP
        self.F = None
        self.CR = None
        self.OF = OF
        
        self.P = None

        self.Aext = None
        self.M_F = None
        self.M_CR = None
        self.S_F = None
        self.S_CR = None
        self.H = H
        self.Asize = None
        self.M_Fhistory = None
        self.M_CRhistory = None
        self.minPopSize = minPopSize
        self.maxPopSize = NP
        self.pMin = None
        self.pMax = None


    def getRandomInd(self, array, toRemove):
        popCopy = array[:]
        for i in toRemove:
            popCopy.remove(i)
        return random.choice(popCopy)

    def pickBests(self, size, id):
        popCopy = self.P[:]
        popCopy.remove(id)
        return sorted(popCopy, key=lambda ind: ind.ofv)[:size]

    def euclid(self, u, v):
        sum = 0
        for i in range(len(u)):
            sum += (u[i] - v[i])**2
        return sum**(1/2)

    def resizeAext(self):
        copy = sorted(self.Aext[:], key=lambda ind: ind.ofv)
        self.Aext = copy[:self.NP]

    def resize(self, array, size):
        copy = sorted(array[:], key=lambda ind: ind.ofv)
        return copy[:size]


    #original.features, pbestInd.features, xr1.features, xr2.features, Fg)
    def mutation(self, x, pbest, xr1, xr2, F, Fw):
        v = list(range(self.dim))

        for i in range(self.dim):
            v[i] = x[i] + Fw * (pbest[i] - x[i]) + F * (xr1[i] - xr2[i])

        return v

    def crossover(self, original, v, CR):
        u = original[:]

        j = random.randint(0, self.dim)

        for i in range(self.dim):
            if (random.uniform(0, 1) <= CR) or (i == j):
                u[i] = v[i]

        return u

    def bound_constrain(self, original, u):

        for i in range(self.dim):
            if u[i] < self.OF.bounds[i][0]:
                u[i] = (self.OF.bounds[i][0] + original[i]) / 2
            elif u[i] > self.OF.bounds[i][1]:
                u[i] = (self.OF.bounds[i][1] + original[i]) / 2

        return u

    def run(self):

        #initialization
        G = 0
        self.Aext = []
        self.M_F = list(range(self.H))
        self.M_CR = list(range(self.H))
        best = None
        fes = 0

        k = 0
        self.pMin = 0.125
        self.pMax = 0.25

        #fill 
        for i in range(0, self.H - 1):
            self.M_F[i] = 0.5
            self.M_CR[i] = 0.5

        self.M_F[self.H-1] = 0.9
        self.M_CR[self.H-1] = 0.9

        #population initialization
        id = 0
        self.P = [Individual(self.dim, self.OF.bounds, id) for x in range(self.NP)]
        for ind in self.P:
            ind.ofv = self.OF.evaluate(ind.features)
            ind.id = id
            id += 1
            fes += 1
            if best == None or ind.ofv <= best.ofv:
                best = ind

        #maxfes exhaustion
        while fes < self.maxFEs:
            G += 1
            newPop = []
            self.S_CR = []
            self.S_F = []
            wS = []

            #generation iterator
            for i in range(self.NP):

                original = self.P[i]

                r = random.randint(0, self.H -1)
                Fg = cauchy.rvs(loc=self.M_F[r], scale=0.1, size=1)[0]
                while(Fg <= 0):
                    Fg = cauchy.rvs(loc=self.M_CR[r], scale=0.1, size=1)[0]
                if(Fg > 1):
                    Fg = 1

                #CRg = cauchy.rvs(loc=self.M_CR[r], scale=0.1, size=1)[0]
                CRg = np.random.normal(self.M_CR[r], 0.1, 1)[0]
                if(CRg > 1):
                    CRg = 1
                if(CRg < 0):
                    CRg = 0

                fesR = fes/self.maxFEs
                if(fesR < 0.6 and Fg > 0.7):
                    Fg = 0.7

                if(fesR < 0.25 and CRg > 0.7):
                    CRg = 0.7
                
                if(fesR >= 0.25 and fesR < 0.5 and CRg > 0.6):
                    CRg = 0.6

                Psize = round(self.pMin + fesR * (self.pMax - self.pMin))
                if(Psize < 2):
                    Psize = 2
                
                Fw = None
                if (fesR < 0.2):
                    Fw = 0.7 * Fg
                elif (fesR < 0.4):
                    Fw = 0.8 * Fg
                else:
                    Fw = 1.2 * Fg


                pBestArray = self.pickBests(Psize, original)

                #parent selection
                pbestInd = random.choice(pBestArray)

                xr1 = self.getRandomInd(self.P, [original, pbestInd])
                xr2 = self.getRandomInd(list(set().union(self.P, self.Aext)), [original, pbestInd, xr1])

                #mutation
                v = self.mutation(original.features, pbestInd.features, xr1.features, xr2.features, Fg, Fw)

                #crossover
                u = self.crossover(original.features, v, CRg)

                #bound constraining
                u = self.bound_constrain(original.features, u)

                #evaluation
                newInd = Individual(self.dim, self.OF.bounds, original.id)
                newInd.features = u
                newInd.ofv = self.OF.evaluate(u)
                fes += 1

                #selection step
                if newInd.ofv <= original.ofv:
                    newPop.append(newInd)
                    if newInd.ofv <= best.ofv:
                        best = newInd
                    self.S_F.append(Fg)
                    self.S_CR.append(CRg)
                    self.Aext.append(original)
                    wS.append(self.euclid(original.features, newInd.features))
                else :
                    newPop.append(original)

                if fes >= self.maxFEs:
                    return best

                if len(self.Aext) > self.NP:
                    self.resizeAext()

            self.P = newPop

            if len(self.S_F) > 0:
                wSsum = 0
                for i in wS:
                    wSsum += i

                if wSsum != 0.:

                    meanS_F1 = 0
                    meanS_F2 = 0
                    meanS_CR1 = 0
                    meanS_CR2 = 0
    
                    for s in range(len(self.S_F)):
                        meanS_F1 += (wS[s] / wSsum) * self.S_F[s] * self.S_F[s]
                        meanS_F2 += (wS[s] / wSsum) * self.S_F[s]
                        meanS_CR1 += (wS[s] / wSsum) * self.S_CR[s] * self.S_CR[s]
                        meanS_CR2 += (wS[s] / wSsum) * self.S_CR[s]
                    
                    if meanS_F2 != 0.:
                        self.M_F[k] = ((meanS_F1 / meanS_F2) + self.M_F[k]/2.)
                    if meanS_CR2 != 0.:
                        self.M_CR[k] = ((meanS_CR1 / meanS_CR2) + self.M_CR[k]/2.)

    
                    k += 1
                    if k >= self.H - 1:
                        k = 0
                
            self.NP = round(self.maxPopSize - (fes/self.maxFEs) * (self.maxPopSize - self.minPopSize))    
            self.P = self.resize(self.P, self.NP)
            self.resizeAext()

        return best

dim = 10 #dimension size
NP = round(25 * math.log(dim) * math.sqrt(dim)) #population size
maxFEs = 5000 #maximum number of objective function evaluations
H = 5 #archive size
minPopSize = 4

sphere = Sphere(dim) #defined test function
de = DISH(dim, maxFEs, sphere, H, NP, minPopSize)
resp = de.run()
print(resp)
