import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import networkx as nx

def ODE(x,t,A,f,g,params):
    N = x.shape[0]
    dXdt = np.zeros(N)
    for n in range(0,N):
        dXdt[n] = f(x[n],params[:,n])
        for j in range(0,N):
            dXdt[n] += A[n,j]*g(x[n],x[j])
    return dXdt

def LV():
    def f(x,param):
        a = param[0]
        b = param[1]
        return x*(a-b*x)
    def g(x,y):
        return(-x*y)
    return f,g

def MP():
    def f(x,param):
        a = param[0]
        b = param[1]
        return x*(a-b*x)
    def g(x,y):
        return(x*(y**2)*((1+x**2)**(-1)))
    return f,g

def MM():
    def f(x,param):
        return -x
    def g(x,y):
        return (y**2)/(1+y**2)
    return f,g

def SIS():
    def f(x,param):
        return -param[0]*x
    def g(x,y):
        return (1-x)*y
    return f,g

def KUR():
    def f(x,param):
        return param[0]
    def g(x,y):
        return np.sin(y-x)
    return f,g


def WC():
    def f(x,param):
        return -x
    def g(x,y):
        return 1/(1+np.exp(-(y-1)))
    return f,g

def weighted_ba_graph(W):
    N = W.shape[0]
    A = np.zeros((N,N))
    for n in range(0,N):
        for m in range(0,N):
            if W[n,m]==1:
                A[n,m] = np.random.uniform(low=-1,high=1)
    return(A)

def derivative_approx(x,dt):
    dx = np.zeros(x.shape)
    for i in range(0,x.shape[1]):
        for t in range(0,x.shape[0]-1):
            dx[t,i] = (x[t+1,i]-x[t,i])*(1/dt)
    return(dx)


models_names = ['LV','MP','MM','SIS','WC','KUR']
ranks = [[],[],[],[],[],[]]


for N in range(20,420,20):
    print('N' + str(N))
    for trial in range(0,100):
        if np.mod(trial,10)==0:
            print(trial)
        g = nx.barabasi_albert_graph(N,1)
        W = nx.to_numpy_array(g)
        A = weighted_ba_graph(W)
        
        a = np.random.uniform(0.5,1.5,N)
        b = np.random.uniform(0.5,1.5,N)
        LV_params = np.zeros((2,N))
        LV_params[0,:] = a
        LV_params[1,:] = b
        MP_params = np.zeros((2,N))
        MP_params[0,:] = a
        MP_params[1,:] = b
    
        MM_params = np.zeros((1,N))
    
        SIS_params = np.random.uniform(0.5,1.5,N).reshape(1,N)
        WC_params = np.zeros((1,N))
        KUR_params = np.random.uniform(0.5,1.5,N).reshape(1,N)
    
        x0 = np.random.uniform(0,1,N)
        LV_IC = x0
        MP_IC = x0
        MM_IC = np.random.uniform(0,2,N)
        SIS_IC = np.random.uniform(0,0.1,N)
        WC_IC = np.random.uniform(0,10,N)
        KUR_IC = np.random.uniform(0,2*np.pi,N)

        models = [LV,MP,MM,SIS,WC,KUR]
        models_params = [LV_params,MP_params,MM_params,SIS_params,WC_params,KUR_params]
        models_IC = [LV_IC,MP_IC,MM_IC,SIS_IC,WC_IC,KUR_IC]
        
        for idx,model in enumerate(models):
            f,g = model()
            x = sp.integrate.odeint(ODE,models_IC[idx],t,args=(A,f,g,models_params[idx],))
            dx = derivative_approx(x,dt)
            Y = dx-f(x,LV_params)
            r =[]
            for i in range(0,N):
                Y_i = Y[:,i]
                G_i = np.zeros((nT,N))
                for k in range(0,nT):
                    for j in range(0,N):
                        G_i[k,j] = g(x[k,i],x[k,j])
                r.append(np.linalg.matrix_rank(G_i))
            ranks[idx].append(np.median(r))

np.savetxt("ranksNvary.csv", ranks, delimiter=",")