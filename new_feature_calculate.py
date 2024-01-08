import numpy as np
from itertools import permutations
import networkx as nx
def calculateSecondOrderDifference(sig):
    nn = np.column_stack((sig[1:-1] - sig[:-2], sig[2:] - sig[1:-1]))
    SX = np.mean(nn[:, 0]**2)
    SY = np.mean(nn[:, 1]**2)
    SXY = np.mean(nn[:, 0] * nn[:, 1])
    D = np.sqrt(np.abs(SX + SY - 4 * (SX * SY - SXY**2)))
    a = 1.7321 * np.sqrt(SX + SY + D)
    b = 1.7321 * np.sqrt(SX + SY - D)
    Ar = np.pi * a * b
    return Ar

def calculateKatzFD(sig):
     # Computing 'L':
    L = 0
    N = len(sig)
    n = N - 1  # 'n' is the number of steps in the waveform.
    for i in range(N - 1):
        aux = np.sqrt(1 + (sig[i] - sig[i+1])**2)
        L += aux

    # Computing 'd':
    dist = [np.sqrt((1 - i)**2 + (sig[0] - sig[i])**2) for i in range(1, N)]
    d = max(dist)

    # Computing of KFD:
    KatzFD = np.log10(n) / (np.log10(n) + np.log10(d/L))
    
    return KatzFD

def func_SRP(sig, m): 
    T = len(sig) - m + 1
    # print(T)
    mh = [np.argsort(sig[i:i+m]) for i in range(T)]
    mh = np.array(mh)
    
    # Generate all possible ordinal patterns for embedding dimension m
    symbs = list(permutations(range(m)))
    
    # Initialize the SRP matrix
    SRP = np.zeros((T, T))
    for symb in symbs:
        loc = np.all(mh == np.array(symb), axis=1)
        A = np.outer(loc, loc)
        SRP += A
    # print("SRP")
    return SRP

def RPcla(sig, m, eps):
    #y i sthe time series to compute the classical recurrence matrix
    #eps is 10 in the paper
    T = len(sig)
    # Compute the m-history
    ym = np.array([sig[k:T-m+k+1] for k in range(m)]).T
    # Initialize the recurrence matrix
    Reclas = np.zeros((T-m+1, T-m+1))
    # print("T-m+1=",T-m+1)
    # print("ymshape=",ym.shape[0])
    for i in range(T-m+1):
        xm_repeated = np.tile(ym[i, :], (T-m+1, 1))
        aa = np.max(np.abs(ym - xm_repeated), axis=1) < eps
        Reclas[:, i] = aa

    # We do not consider the self-recurrences
    results = Reclas - np.eye(T-m+1)
    # print("cla")
    return results

def calculateGraph(AC,SRP):
    G = nx.Graph(AC * SRP)
    return G

def calculateMeanNetDegree(G):

    deg = np.mean(list(dict(G.degree()).values()))  
    # print("deg")
    return deg

def calculateMeanNetBetweeness(G):

    betw = np.mean(list(nx.betweenness_centrality(G).values())) 
    return betw

def calculateMeanNetCloseness(G):

    clos = np.mean(list(nx.closeness_centrality(G).values())) 
    return clos

def calculateNetwork(sig):
    AC = RPcla(sig,3,10)
    SRP = func_SRP(sig,3)
    G = nx.Graph(AC * SRP)

    deg = calculateMeanNetDegree(G)
    betw = calculateMeanNetBetweeness(G)
    clos = calculateMeanNetCloseness(G)

    AllFeature = [deg,betw,clos]
    return AllFeature