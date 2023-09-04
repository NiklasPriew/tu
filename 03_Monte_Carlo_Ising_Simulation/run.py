from cmath import exp
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from numba import jit
from scipy.ndimage import convolve, generate_binary_structure

Inn=np.array([1,-1,0,0])
Jnn=np.array([0,0,1,-1])
@jit(nopython=True)
def init_mag(L):
    #print(L)
    mag = np.ones((L,L))

    for i in range(0,L):
        for j in range(0,L):
            rand = random.randint(-1,1)
            if rand < 0.3:
                mag[i,j] = mag[i,j] * (-1)
    return mag
@jit(nopython=True)
def calcE(mag,i,j):
    Energy = 0
    for k in range(0,3):
            inew = i +Inn[k]
            jnew = j + Jnn[k]
            if inew < 0:
                inew = L-1
            if inew >= L:
                inew = 0
            if jnew < 0:
                jnew = L-1
            if jnew >= L:
                jnew = 0
            Energy = Energy + -J * mag[i][j] * mag[inew][jnew]
            #print(Energy)
    return Energy
@jit(nopython=True)
def sweep(mag,L, T,J):
    for x in range(0,L*L):
        i = random.randint(0,L-1)
        j = random.randint(0,L-1)
        
        Energy = calcE(mag,i,j)
        
        mag[i,j] = mag[i,j] * -1

        Energynew = calcE(mag,i,j)
        deltaEnergy = Energynew - Energy
        #print(deltaEnergy)
        #print(T)
        r = np.exp(-(deltaEnergy/T))
        if min(1,r) < np.random.random():
            
            mag[i,j] = mag[i,j] * -1
            #print("noflip")
    return
@jit(nopython=True)
def monte(L,T,J,warmup,mess):
    mag = init_mag(L)
    #print(mag)
    for x in range(1,warmup):
        sweep(mag,L,T,J)
    m = 0
    m2 = 0
    m4 = 0
    #print(mag)
    for x in range(1,mess):
        sweep(mag,L,T,J)
        mtemp = 0
        mtemp2 = 0
        mtemp4 = 0

        for k in range(0,L):
            for l in range(0,L):
                mtemp = mtemp + mag[k,l]
                mtemp2 = mtemp2 + mag[k,l]
                mtemp4 = mtemp4 + mag[k,l]
        
        #m= m+np.abs(mag.sum())/(L*L)
        #print(mtemp)
        m = m+np.abs(mtemp)/(L*L)
        m2 = m
        m4 = m
    #print(mag)

    m = m/mess
    m2 = m2/mess
    m4 = m4/mess
    #print("Magnetisierung")
    #print(m)
    return m,m2,m4

@jit(nopython=True, parallel = True)
def tempchange(L,warmup,mess):
    tempchangem = np.zeros(100)
    tempchangemm = np.zeros(100)
    tempchangemmmm = np.zeros(100)
    T = np.linspace(0.25,5,100)
    for i,x in enumerate(T):
        #print(x)
        tempchangem[i], tempchangemm[i], tempchangemmmm[i] =  monte(L,x,J,warmup,mess)
    return T,tempchangem,tempchangemm, tempchangemmmm

def ising(L,warmup,mess):
    binder4 = np.zeros(100)
    binder8 = np.zeros(100)
    binder16 = np.zeros(100)
    binder32 = np.zeros(100)
    x = 0
    print('4')
    T,tempchangem4, tempchangemm, tempchangemmmm = tempchange(4,warmup,mess)
    for x in range(0,100):
        binder4[x] = 1-1/3*(tempchangemmmm[x]/tempchangemm[x]**2)
    print('8')
    T,tempchangem8, tempchangemm, tempchangemmmm = tempchange(8,warmup,mess)
    for x in range(0,100):
        binder8[x] = 1-1/3*(tempchangemmmm[x]/tempchangemm[x]**2)
    print('16')
    T,tempchangem16, tempchangemm, tempchangemmmm = tempchange(16,warmup,mess)
    for x in range(0,100):
        binder16[x] = 1-1/3*(tempchangemmmm[x]/tempchangemm[x]**2)
    print('32')
    T,tempchangem32, tempchangemm, tempchangemmmm = tempchange(32,warmup,mess)

    for x in range(0,100):
        binder32[x] = 1-1/3*(tempchangemmmm[x]/tempchangemm[x]**2)

    plt.figure()
    plt.scatter(T,binder4, label = 'Binder Kummulante 4x4' )
    plt.scatter(T,binder8, label = 'Binder Kummulante 8x8' )
    plt.scatter(T,binder16, label = 'Binder Kummulante 16x16' )
    plt.scatter(T,binder32, label = 'Binder Kummulante 32x32' )
    plt.xlabel('Temperature')
    plt.ylabel('Binder Kummulante') 
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(T,tempchangem4, label = '4x4 lattice' )
    plt.scatter(T,tempchangem8, label = '8x8 lattice' )
    plt.scatter(T,tempchangem16, label = '16x16 lattice' )
    plt.scatter(T,tempchangem32, label = '32x32 lattice' )
    plt.xlabel('Temperature')
    plt.ylabel('<M>') 
    plt.legend()
    plt.show()
#init_mag(L)
#print(mag)
#tempchange(L,Energy,warmup,sweeps)
if __name__ == "__main__":
    L = 8
    J = 1
    warmup = 1000
    sweeps = 10000
    ising(L,warmup, sweeps)