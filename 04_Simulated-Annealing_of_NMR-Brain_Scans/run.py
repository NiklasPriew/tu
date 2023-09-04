from cmath import exp
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import numba
from numba import jit
from scipy.ndimage import convolve, generate_binary_structure


value = np.array([0,30,426,602,1223,167])
sigma = np.array([0,30,59,102,307,69])

@jit(nopython=True)
def init_mag(x,y):
    #print(L)
    #mag = np.random.random((x,y))*5+1
    #mag = mag.astype(int)
    mag = np.random.randint(1,6,(x,y))
    print(mag)
    return mag

@jit(nopython=True)
def calcE(mag,f,i,j,J):
    A = 254
    B = 333
    Energy = 0
    E1 = 0
    E2 = 0
    Inn=np.array([1,-1,0,0])
    Jnn=np.array([0,0,1,-1])
    for k in range(0,3):
        inew = i + Inn[k]
        jnew = j + Jnn[k]
        if inew < 0:
            inew = A-1
        if inew >= A:
            inew = 0
        if jnew < 0:
            jnew = B-1
        if jnew >= B:
            jnew = 0
        if(mag[i,j] == mag[inew,jnew]):
            E1 = E1+1
    E2 = (f[i*333+j] - value[mag[i,j]])**2/(2*sigma[mag[i,j]]**2) + np.log(sigma[mag[i,j]])
            #print(Energy)
    Energy = -J *E1+E2
    return Energy
@jit(nopython=True)  
def sweep(mag,a,b,f, T,J):
    for x in range(0,a*b):
        i = random.randint(0,a-1)
        j = random.randint(0,b-1)
        
        
        Energy = calcE(mag,f,i,j,J)
        magprev = mag[i,j]
        mag[i,j] = np.random.randint(1,6)

        Energynew = calcE(mag,f,i,j,J)
        deltaEnergy = Energynew - Energy

        r = np.exp(-(deltaEnergy/T))
        #if min(1,r) <= np.random.random():
        #    mag[i,j] = magprev
            #print("noflip")
        if not np.random.random() < min(1,r):
            mag[i,j] = magprev
        
            
    return mag
@jit(nopython=True)  
def simann(a,b,T,Tf,J,warmup,lines):
    mag = init_mag(a,b)
    f = lines
    l = 1.01
    for x in range(0,warmup):
        sweep(mag,a,b,f,T,J)
        #print(x)

    while Tf < T:
        for x in range(0,1):
            sweep(mag,a,b,f,T,J)
        T = T/l
        #print(T)
    #draw(mag,a,b)
    return mag

def draw(mag,a,b):
    plt.imshow(mag)
    plt.colorbar()
    # show plot
    plt.show()


a = 254
b = 333
J = 2
T = 1
Tf = 0.001
warmup = 1000
sweeps = 1000
intline = np.zeros((a*b)-1)
cintline = np.zeros((a*b)-1)
f = open("Daten.dat", "r")
lines = f.readlines()
c = open("CorrectSegImage.dat", "r")
clines = c.readlines()


print(numba.int64(lines[5]))
for x in range(0,(a*b)-1):
    intline[x] = numba.int64(lines[x])
    cintline[x] = numba.int64(clines[x])

mag = simann(a,b,T,Tf,J,warmup,intline)
mag1d = np.zeros((a-1)*(b-1))

counttype= np.zeros(5)
counttypeerror= np.zeros(5)

for x in range(0,a-1):
    for y in range(0,b-1):
        print(x)
        print(y)
        mag1d[x+y] = mag[x-1][y-1]
mag1d = mag.flatten()
a_file = open("test.txt", "w")
np.savetxt(a_file, mag1d)

a_file.close()

print(clines[3])
for x in range(0, (a-1)*(b-1)):
    if cintline[x] == 1:
        counttype[0] = counttype[0] + 1
        if mag1d[x] != 1:
            counttypeerror[0] = counttypeerror[0] + 1
    if cintline[x] == 2:
        counttype[1] = counttype[1] + 1
        if mag1d[x] != 2:
            counttypeerror[1] = counttypeerror[1] + 1
    if cintline[x] == 3:
        counttype[2] = counttype[2] + 1
        if mag1d[x] != 3:
            counttypeerror[2] = counttypeerror[2] + 1
    if cintline[x] == 4:
        counttype[3] = counttype[3] + 1
        if mag1d[x] != 4:
            counttypeerror[3] = counttypeerror[3] + 1
    if cintline[x] == 5:
        counttype[4] = counttype[4] + 1
        if mag1d[x] != 5:
            counttypeerror[4] = counttypeerror[4] + 1

print(counttype[1])
print(counttypeerror[1])
print("Fehler")

print("Typ 1")
print(counttypeerror[0]/counttype[0])
print("Typ 2")
print(counttypeerror[1]/counttype[1])
print("Typ 3")
print(counttypeerror[2]/counttype[2])
print("Typ 4")
print(counttypeerror[3]/counttype[3])
print("Typ 5")
print(counttypeerror[4]/counttype[4])

draw(mag,a,b)