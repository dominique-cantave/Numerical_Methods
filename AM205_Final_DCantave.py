
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from math import *
import warnings
import time
from operator import attrgetter
from copy import deepcopy
warnings.filterwarnings("ignore", category=RuntimeWarning)



""" SIMULATING DATA FROM ODE SYSTEM 
    variables:
        a,b,p (k): parameters from Huang et al. 2007 paper
        it,ts: number of initial points and timesteps to calculate
        r,h: range of x1,x2 values and step size

"""

#auto-induction rate; cross-inhibition rate; decay rate
a=1; b=1; p=1;
#num initial points; num time steps
it=20; ts=600
#range of values to consider
r=[0,4]
#time-step h
h=0.01

#differential equations
def dx1(x1,x2,n,th):
    """
        dx1: differential equation from Huang et al. 2007 paper
            parameters: x1, x2, n, theta
            returns: derivative in x1 direction
    """
    return a*x1**n/(th**n+x1**n)+b*th**n/(th**n+x2**n)-p*x1
    
def dx2(x1,x2,n,th):
    """
        dx2: differential equation from Huang et al. 2007 paper
            parameters: x1, x2, n, theta
            returns: derivative in x2 direction
    """
    return a*x2**n/(th**n+x2**n)+b*th**n/(th**n+x1**n)-p*x2


# run system according to ODEs
def find_path(init, n, theta):
    """
        find_path: ODE trajectories using Taylor series approximation
            parameters: initial values, n, theta
            returns: trajectory from initial point to ts timesteps later
    """
    path=np.zeros((ts,2))

    path[0]=init
    for k in range(1,ts):
        # f(x_(k+1)=f(x_k)+hf'(x)) from Taylor approximation
        path[k,0]=path[k-1,0]+h*dx1(path[k-1,0],path[k-1,1],n,theta)
        path[k,1]=path[k-1,1]+h*dx2(path[k-1,0],path[k-1,1],n,theta)
    return path



""" GRAPHING TRAJECTORIES AND ADDING NOISE
    variables:
        n_theta: set of n and theta combinations to use
        inits: set of it initial points to use
    
    for each n/theta combination:
        calculate exact trajectories for each initial point (stored in X_ob)
        add Gaussian noise to all points except initial (stored in Xa)
        graph original and noisy trajectories
        add Xa to X - list of datasets for each n/theta combination
"""

# set of n and theta combination to set 
n_theta=np.array([[2,0.2],[6,0.4],[4,1.2],[8,1.6]])
# initial values
inits=np.random.uniform(*r,[it,2])

# initialize plot information
fig, ax=plt.subplots(2,2,figsize=(10,10))
k=0

# initialize datasets
X=np.empty((4,it,ts,2))

# for each set of n and theta combination
for n,th in n_theta:
    #initialize trajectory matrices
    X_ob=np.zeros((it,ts,2))
    Xa=np.zeros((it,ts,2))
    
    # find path and add Gaussian noise
    for g in range(it):
        X_ob[g]=find_path(inits[g],n,th)
        Xa[g]=X_ob[g]+np.random.normal(0,0.015,[ts,2]); Xa[g,0]=X_ob[g,0]

        # graph trajectories and initial/final points
        if g==0:
            ax[floor(k/2),k%2].plot(X_ob[g,:,0],X_ob[g,:,1],color='b',label='original',zorder=1)
            ax[floor(k/2),k%2].plot(Xa[g,:,0],Xa[g,:,1],color='r',label='with noise',zorder=0)
        else:
            ax[floor(k/2),k%2].plot(X_ob[g,:,0],X_ob[g,:,1],color='b',zorder=1)
            ax[floor(k/2),k%2].plot(Xa[g,:,0],Xa[g,:,1],color='r',zorder=0)

        ax[floor(k/2),k%2].scatter(X_ob[g,0,0],X_ob[g,0,1],color='m',zorder=10)
        ax[floor(k/2),k%2].scatter(X_ob[g,-1,0],X_ob[g,-1,1],color='y',zorder=10)

    # other graphing parameters
    ax[floor(k/2),k%2].set_xlabel('x1=GATA1')
    ax[floor(k/2),k%2].set_ylabel('x2=PU.1')
    ax[floor(k/2),k%2].set_title('n={},th={}'.format(n,th))
    ax[floor(k/2),k%2].legend()
    ax[floor(k/2),k%2].set_xlim(0,4)
    ax[floor(k/2),k%2].set_ylim(0,4)
    
    # store trajectory
    X[k]=Xa
    k+=1
    
plt.tight_layout()
print("Printing the trajectories of {} initial points using parameter sets:".format(it))
print("\n[n theta]\n{}\n{}\n{}\n{}".format(n_theta[0],n_theta[1],n_theta[2],n_theta[3]))
fig.suptitle("Trajectories with Noise")
fig.savefig('noisy.png')
plt.show(fig)



""" FINDING BIFURCATION POINTS 
    repeat trajectory calculation and graphing for incrementally
    increasing n and theta values
    
"""
# graphing information
bi_fig, bi_ax=plt.subplots(5,4,figsize=(15,15))

for i in range(1,5):
    for j in range(5):
        # define n and theta incrementally
        n=2.*i
        th=2*j/5
        
        #initialize trajectory matrices
        X_ob=np.zeros((it,ts,2))

        # graph trajectories
        for g in range(it):
            X_ob[g]=find_path(inits[g],n,th)
            if g==0:
                bi_ax[j,i-1].plot(X_ob[g,:,0],X_ob[g,:,1],color='b',label='original')
            else:
                bi_ax[j,i-1].plot(X_ob[g,:,0],X_ob[g,:,1],color='b')
                
            bi_ax[j,i-1].scatter(X_ob[g,0,0],X_ob[g,0,1],color='m',zorder=10)
            bi_ax[j,i-1].scatter(X_ob[g,-1,0],X_ob[g,-1,1],color='y',zorder=10)
            
        # other graphing parameters
        bi_ax[j,i-1].set_xlabel('x1=GATA1')
        bi_ax[j,i-1].set_ylabel('x2=PU.1')
        bi_ax[j,i-1].set_title('n={},th={}'.format(n,th))
        bi_ax[j,i-1].legend()
        bi_ax[j,i-1].set_xlim(0,4)
        bi_ax[j,i-1].set_ylim(0,4)
        
print("\n\nBifurcation Analysis using parameter increments")
bi_fig.suptitle("Bifurcation analysis")
plt.tight_layout()
plt.show(bi_fig)
bi_fig.savefig('bifurcation.png')



""" GRAPHING PARAMETER SPACE AND ERROR 
    varibles:
        ns, thetas: parameter values to calculate error across parameter space
        
    for each n/theta combination
        calculate parameter error for all ns, thetas combinations (stored in params_a)
        add params_a to params
"""
# list of n and theta values within parameter space
ns=np.arange(0,10,0.2)
thetas=np.arange(0,2,0.1)

def find_error(n,th,X):
    """
        find_error: calculate error between known trajectory and potential trajectory
            parameters: n, theta, known trajectories
                calculate potential trajectory from known initial and n/theta
                calculate error for each trajecory (norm squared difference)
                add to total error measure (sum over all initial points)
                    (if n/theta is infeasible, highly penalize)
            returns: error
    """
    err=0
    for k in range(len(X)):
        init=X[k,0]
        traj=find_path(init,n,th)
        add=np.sum(np.linalg.norm(traj-X[k],axis=1)**2)
        if str(add)=='nan':
            err+=1e6
            continue

        err += add
    
    return err

# store errors for each n/theta set
params=np.zeros((4,len(thetas),len(ns)))
print("\n\nCalculating errors throughout the parameter space for each set of parameters")
# for each known n/theta combination
for k in range(4): 
    params_a=np.zeros((len(thetas),len(ns)))
    # for all possible ns and thetas in parameter space
    for i in range(len(thetas)):
        for j in range(len(ns)):
            # calculate error
            params_a[i,j]=find_error(ns[j],thetas[i],X[k])
            
    params[k]=params_a
    print("parameter set {} complete!".format(n_theta[k]))



# graph on meshgrid
nn,tt=np.meshgrid(ns,thetas)

fig = plt.figure(figsize=(10,10))

for i in range(4):
    ax = fig.add_subplot(2,2,i+1, projection='3d')
    ax.plot_surface(nn,tt,params[i],cmap='viridis')
    ax.set_xlabel('n')
    ax.set_ylabel('theta')
    ax.set_zlabel('error')
    ax.set_title('n={}, th={}'.format(n_theta[i,0],n_theta[i,1]))
    
print("Graph of the error landscape")
plt.tight_layout()
fig.savefig('parameterspace_1.png')
plt.show(fig)



# focusing only on combinations with small error measures

fig = plt.figure(figsize=(10,10))
for i in range(4):
    t2,n2=np.where(params[i]<350)
    tt,nn=np.meshgrid(t2,n2)
    new_els=params[i,tt,nn]

    ax = fig.add_subplot(2,2,i+1, projection='3d')
    ax.plot_surface(ns[nn],thetas[tt],new_els,cmap='viridis')
    ax.set_xlabel('n')
    ax.set_ylabel('theta')
    ax.set_zlabel('error')
    ax.set_title('n={}, th={}'.format(n_theta[i,0],n_theta[i,1]))
    ax.set_zlim3d(0,400)
    ax.set_xlim(0,10)
    ax.set_ylim(0,2)
    
print("Close view of the error landscape around the minimum")
plt.tight_layout()
fig.savefig('parameterspace_2.png')
plt.show(fig)



# calculate actual error between correct parameter values and noisy data
print("actual error between correct parameter values and noisy data")
for i in range(4):
    print('n={}, th={}, err={}'.format(*n_theta[i],find_error(*n_theta[i],X[i])))



""" STEEPEST DESCENT ALGORITHM 
    variables:
        hn: step size for df/dn
        ht: step size for df/dtheta
"""


hn=0.001
ht=0.001

def error_grad(n,th,X):
    """
        error_grad: gradient of error function using finite difference approx.
            parameters: n, theta, known trajectories
                if n and theta are within the feasible region, calculate gradient with backwards difference
                otherwise redirect back to feasible region
            returns: gradient
    """
    # if n-hn is feasible (meaning between 0 and 10)
    if n-hn>0 and n<=10:
        # calculate finite difference approx to gradient
        dfdn=(find_error(n,th,X)-find_error(n-hn,th,X))/(hn)
    else:
        dfdn=10*n
    
    if th-ht>0 and th<=2:
        dfdt=(find_error(n,th,X)-find_error(n,th-ht,X))/(ht)
    else:
        dfdt=10*th
        
    return np.array([dfdn,dfdt])

def eta_err(eta,n,th,X,sk):
    """
        eta_err: used to determine eta to minimize f(xk+eta*sk)
            parameters: eta, n, theta, known trajectories, sk
            returns: f(xk+eta*sk)
    """
    ans=find_error(n+eta*sk[0],th+eta*sk[1],X)
    return ans

def steepest_descent(x0,X):
    """
        steepest_descent: steepest descent algorithm to minimize error
            parameters: initial guess, known trajectories
                find sk=-grad(f(x))
                'find etak by minimizing f(xk+eta*sk)
                x(k+1)=xk+etak*sk
                iterate until xk converges or 100 iterations achieved
            returns: minimum value and error
    """
    xk=np.array(x0,dtype=float)
    N=100
    old_x=np.inf
    k=0
    while abs(np.linalg.norm(xk-old_x))>0.5:
        k+=1
        old_x=xk.copy()
        sk=-error_grad(*xk,X)
        res=opt.minimize(eta_err,0.1,args=(*xk,X,sk), bounds=((0,None),), options={'maxfun': 50})
        etak=float(res.x)
        
        xk+=etak*sk
        if k == N:
            break
    return xk,find_error(*xk,X)



""" RUNNING STEEPEST DESCENT 
    for each n/theta combination:
        run 10 iterations of steepest descent
        retain the best parameter set (with the lowest error)
        calculate distance from actual parameters and total run time
"""
print("RUNNING STEEPEST DESCENT ALGORITHM")
for j in range(4):
    start=time.clock()
    m=10 # number of iterations
    best_err_sd=np.inf
    print('\n\nactual value: ',n_theta[j])
    for i in range(m):
        x0=np.random.uniform([0,0],[10,2],2)
        n_th, err = steepest_descent(x0,X[j])
        if err<best_err_sd:
            best_err_sd=err
            best_nth_sd=n_th
        print('iteration ',i,', initial: ',x0,', final: ',n_th,err)
    end=time.clock()
    print('\nlowest error: ', best_err_sd)
    print('[n,theta]: ', best_nth_sd)
    print('squared dist: ', np.linalg.norm(best_nth_sd-n_theta[j])**2)
    print('Run-time: ', end-start)



""" BFGS ALGORITHM 
            
    for each n/theta combination:
        run 10 iterations of BFGS
        retain the best parameter set (with the lowest error)
        calculate distance from actual parameters and total run time
"""


def del_B(y,s,B):
    """
        del_B: calculate deltaBk in BFGS algorithm
            parameters: yk, sk, Bk
            returns: delta Bk according to formula in write-up
    """
    y=y[:,np.newaxis]
    s=s[:,np.newaxis]
    return (y@y.T)/(y.T@s)-(B@s@s.T@B)/(s.T@B@s)

def BFGS(x0,X):
    """
        BFGS: run BFGS algorithm
            parameters: initial guess, known trajectories
                calculate sk by solving Bk sk=grad(f(xk))
                calculate x(k+1)=xk+sk
                calculate y(k+1)=grad(f(x(k+1)))-grad(f(xk))
                calculate B(k+1)=Bk+deltaBk
                continue until convergence or 100 iterations achieved
            returns: minimized value and error
    """
                
    xk=np.array(x0,dtype=float)
    Bk=np.identity(2)
    N=100
    old_x=np.inf
    k=0
    
    while abs(np.linalg.norm(xk-old_x))>0.5:
        k+=1
        old_x=xk.copy()
        
        sk=np.linalg.solve(Bk,-error_grad(*xk,X))
        xk+=sk
        
        yk=error_grad(*xk,X)-error_grad(*old_x,X)
        Bk+=del_B(yk,sk,Bk)
        
        if k == N:
            break
    return xk,find_error(*xk,X)



""" RUNNING BFGS
    for each n/theta combination:
        run 10 iterations of BFGS
        retain the best parameter set (with the lowest error)
        calculate distance from actual parameters and total run time
"""
print("RUNNING BFGS ALGORITHM")
for j in range(4):
    start=time.clock()
    m=10 # number of iterations
    best_err_bfgs=np.inf
    print('\n\nactual value: ',n_theta[j])
    for i in range(m):
        x0=np.random.uniform([0,0],[10,2],2)
        n_th, err = BFGS(x0,X[j])
        if err<best_err_bfgs:
            best_err_bfgs=err
            best_nth_bfgs=n_th
        print('iteration: ',i,'initial: ',x0,'final: ',n_th,err)
    end=time.clock()
    print('\nlowest error: ', best_err_bfgs)
    print('[n,theta]: ', best_nth_bfgs)
    print('squared dist: ', np.linalg.norm(best_nth_bfgs-n_theta[j])**2)
    print('Run-time: ', end-start)



""" GENETIC ALGORITHM """

class Individual():
    """
        Individual: set of parameters [n,theta]
            attributes: parameters, fitness (-error)
    """
    def __init__(self,n,th):
        self.fitness = None
        self.pars = np.array([n,th])    

def init_pop(n):
    """
        init_pop: create initial population
            parameters: population size
                create random values of n and theta
                add n individuals to population list with n/theta values
            returns: population list
    """
    nt=np.random.uniform([0,0],[10,2],[n,2])
    pop=[]
    for i in range(n):
        pop.append(Individual(*nt[i]))
    
    return pop

def selection(pop, num_tot, num_best, num_worst,tourn_size):
    """
        selection: select members of new generation
            parameters: pop list, pop size, number of best individuals to choose
                    number of worst individuals to choose, tournament size
                add num_best individuals with highest fitness
                add num_worst individuals with lowest fitness
                conduct tournament, choosing best of three random individuals, until filled
            returns: population
    """
    chosen=[]
    chosen.extend(sorted(pop, key=attrgetter("fitness"), reverse=True)[:num_best])
    chosen.extend(sorted(pop,key=attrgetter("fitness"))[:num_worst])
    
    for i in range(num_best+num_worst,num_tot):
        tourn = [np.random.choice(pop) for i in range(tourn_size)]
        chosen.append(max(tourn, key=attrgetter("fitness")))
    
    return chosen

def mutation(ind):
    """
        mutatation: mutate n and theta parameters
            parameters: individual
                add N(0,0.01) noise to each parameter
                clear fitness
            returns: individual
    """
    ind.pars = ind.pars+np.random.normal(0,0.01,2)
    ind.fitness = None

    return ind

def mating(ind1,ind2):
    """
        mating: cross over two individuals to swap parameters
            parameters: 2 individuals
                swap n and theta values
                clear fitnesses
            returns: 2 new individuals
    """
    a=ind1.pars.copy()
    b=ind2.pars.copy()
    temp=a.copy()
    a[0]=b[0]
    b[0]=temp[0]
    ind1.pars=a; ind1.fitness=None
    ind2.pars=b; ind2.fitness=None
    
    return ind1,ind2

def fitness(population,X):
    """
        fitness: calculate fitness of whole population
            parameters: population list, known trajectories
                calculate fitness=-error for all individuals without one
            returns: populations with fitnesses
    """
    for ind in population:
        if ind.fitness == None:
            ind.fitness=-find_error(*ind.pars,X)
    return population

def GA(NGEN,POP,CXPB,MUTPB,X):
    """
        GA: run genetic algorithm
            parameters: number of generations, population size, cross over probability,
                    mutation probability, known trajectories
                initialize population and calculate fitnesses
                for NGEN generations:
                    create offspring from original population
                    crossover and mate offspring
                    calculate offspring fitnesses
                    select new population from prev. generation and offspring
                    
            returns: best in population and error
    """
    population=init_pop(POP)
    population=fitness(population,X)
    
    for g in range(NGEN):
        
        offspring=deepcopy(population)
        for i in range(1,len(offspring),2):
            if np.random.rand() < CXPB:
                offspring[i-1],offspring[i] = mating(offspring[i-1],offspring[i])
            
        for i in range(len(offspring)):
            if np.random.random() < MUTPB:
                offspring[i] = mutation(offspring[i])
            
        offspring=fitness(offspring,X)
        
        population=selection(population+offspring, POP, 10, 5, 3)
        
    
    best=max(population, key=attrgetter("fitness"))
        
    return best.pars,-best.fitness



""" RUN GENETIC ALGORITHM
    variables: number of generations, population size, cross over probability,
            mutation probability
            
    for each n/theta combination:
        run 10 iterations of GA
        retain best final parameter set and error over all runs
        calculate distance to known answer
        calculate run time
"""
print("RUNNING GENETIC ALGORITHM")
NGEN = 50
POP = 50
CXPB = 0.3       # rate of crossover
MUTPB = 0.5      # rate of mutation
print("NGEN = {}, POP = {}, CXPB = {}, MUTPB = {}".format(NGEN,POP,CXPB,MUTPB))
for k in range(4):
    start1=time.clock()
    m=10
    best_err_GA=np.inf
    
    print('\n\nactual values: ',n_theta[k])
    for i in range(m):
        start=time.clock()
        final_param, err = GA(NGEN,POP,CXPB,MUTPB,X[k])
        if err<best_err_GA:
            best_err_GA=err
            best_nth_GA=final_param
        end=time.clock()
        print('final params: ', final_param, 'best err: ', err)
        print('Run-time: ', end-start)
        
    end1=time.clock()
    print('\nlowest error: ', best_err_GA)
    print('[n,theta]: ', best_nth_GA)
    print('squared dist: ', np.linalg.norm(best_nth_GA-n_theta[k])**2)
    print('Run-time: ', end1-start1)



""" RUN GA + SD
    variables: number of generations, population size, cross over probability,
            mutation probability
            
    for each n/theta combination:
        run 10 iterations of GA
        retain best final parameter set and error over all runs
        run steepest descent with best parameters as initial value
        find final value and error with steepest descent
        calculate distance to known answer
        calculate run time
"""
print("RUNNING GENETIC ALGORITHM WITH STEEPEST DESCENT")
NGEN = 30
POP = 25
CXPB = 0.3       # rate of crossover
MUTPB = 0.5      # rate of mutation
print("NGEN = {}, POP = {}, CXPB = {}, MUTPB = {}".format(NGEN,POP,CXPB,MUTPB))
for k in range(4):
    start1=time.clock()
    m=10
    best_err=np.inf
    
    print('\n\nactual values: ',n_theta[k])
    for i in range(m):
        prog_param, err = GA(NGEN,POP,CXPB,MUTPB,X[k])
        if err<best_err:
            best_err=err
            best_nth=prog_param
        print('final params: ', prog_param, 'best err: ', err)
    
    print('\ninit params: ', best_nth)
    n_th, err = steepest_descent(best_nth,X[k])
    end1=time.clock()
    
    print('final error: ', err)
    print('[n,theta]: ', n_th)
    print('squared dist: ', np.linalg.norm(n_th-n_theta[k])**2)
    print('Run-time: ', end1-start1)

