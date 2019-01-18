
# coding: utf-8

# In[2]:


import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import imageio as io
import math
import pandas as pd


# ## Question 1
# ### Polynomial approximation of the gamma function.

# #### a
# The gamma function is defined as:
# $$\Gamma(x) = \int_0^\infty t^{x-1}e^{-t} dt$$
# and satisfies $(n - 1)! = \Gamma(n)$ for integers $n$. Construct an approximation to the gamma function by finding the polynomial Lagrange interpolant of the
# following points
# 
# $\;\;n\;\;$ | 1 | 2 | 3 | 4 | 5
# 
# $\Gamma(n)$ | 1 | 1 | 2 | 6 | 24
# 
# Write the interpolant as $g(x) = \sum_{k=0}^4 g_kx^k$
# , and include the values of the coefficients $g_k$
# in your solutions.

# In[68]:


# creating the interpolation points for a and b
print("-"*40)
print("QUESTION 1: Polynomial approximation of the gamma function.")
n = list(range(1,6))
gamma = spec.gamma(n)
logg = [math.log(x) for x in gamma]

# finding the g(x) coefficients
# I used the Lagrange polynomial method from lecture, using a formula for the arrangement of L coeffs
# (x-a)(x-b)(x-c)(x-d)=abcd-(abc+abd+acd+bcd)x+(ab+ac+ad+bc+bd+cd)x^2-(a+b+c+d)x^4
# then g=sum(y_k*L_k)
g=np.zeros(5)
for k in range(len(n)):
    temp = [x for x in n if x != n[k]]
    L=1
    L=np.array([temp[0]*temp[1]*temp[2]*temp[3],-temp[0]*temp[1]*temp[2]-temp[0]*temp[1]*temp[3]-temp[0]*temp[2]*temp[3]-temp[1]*temp[2]*temp[3], temp[0]*temp[1]+temp[0]*temp[2]+temp[1]*temp[2]+temp[0]*temp[3]+temp[1]*temp[3]+temp[2]*temp[3],-temp[0]-temp[1]-temp[2]-temp[3],1])
    div=(n[k]-temp[0])*(n[k]-temp[1])*(n[k]-temp[2])*(n[k]-temp[3])
    L=L/div
    g+=gamma[k]*L

print("-"*40)
print("PART A: Write the interpolant g(x) and and include the values of the coefficients in your solution\n")
print('g(x) coefficients=')
print(g)


# #### b
# Construct a second approximation to the gamma function by first calculating
# the fourth order polynomial $p(x)$ that interpolates the points $(n, log(\Gamma(n)))$ for
# $n = 1, 2, 3, 4, 5$. Then define the approximation by $h(x) = \exp(p(x))$.

# In[31]:


# same procedure as part a with the new log y-values
h=np.zeros(5)
for k in range(len(n)):
    temp = [x for x in n if x != n[k]]
    L=np.array([temp[0]*temp[1]*temp[2]*temp[3],-temp[0]*temp[1]*temp[2]-temp[0]*temp[1]*temp[3]-temp[0]*temp[2]*temp[3]-temp[1]*temp[2]*temp[3], temp[0]*temp[1]+temp[0]*temp[2]+temp[1]*temp[2]+temp[0]*temp[3]+temp[1]*temp[3]+temp[2]*temp[3],-temp[0]-temp[1]-temp[2]-temp[3],1])
    div=(n[k]-temp[0])*(n[k]-temp[1])*(n[k]-temp[2])*(n[k]-temp[3])
    L=L/div
    h+=logg[k]*L
    
print("PART B: calculating the fourth order polynomial  p(x)  that interpolates the points  (n,log(Γ(n))h")
print('\nh(x) coefficients=')
print(h)


# #### c
# Plot $\Gamma(x)$, $g(x)$, and $h(x)$ on the interval $1 \le x \le 5$.

# In[69]:


# starting with a list of x values
x=np.linspace(1,5,1001)
y=[]
z=[]

# apply the g(x) and h(x) formulas
for i in x:
    s=0
    t=0
    
    # for each x, sum the coefficients and the powers of x
    for j in range(len(g)):
        s+=(i**j)*g[j]
        t+=(i**j)*h[j]
    y.append(s)
    z.append(np.exp(t))
    
# plot
print("PART C: Plot  Γ(x) ,  g(x) , and  h(x)  on the interval  1≤x≤5")
plt.plot(x,spec.gamma(x),label='Gamma(x)')
plt.plot(x,y,label='g(x)')
plt.plot(x,z,label='h(x)')
plt.legend()
plt.savefig("1c.png")


# #### d
# Calculate the maximum relative error between $\Gamma(x)$ and $g(x)$ on the interval
# $1 \le x \le 5$, accurate to at least three significant figures. Repeat this for $\Gamma(x)$ and $h(x)$. Which of the two approximations is more accurate?

# In[36]:


# d

# taking the relative errors of each function for all x values and finding the maximum
maxg = np.amax(abs(spec.gamma(x)-y)/spec.gamma(x))
maxh = np.amax(abs(spec.gamma(x)-z)/spec.gamma(x))

#plotting for visualization
plt.plot(x,abs(spec.gamma(x)-y)/spec.gamma(x),label='g(x)')
plt.plot(x,abs(spec.gamma(x)-z)/spec.gamma(x),label='h(x)')
plt.legend()
print("PART D:  Calculate the maximum relative error between  Γ(x)  and  g(x)  on the interval  1≤x≤5")
print('\nmax(|f(x)-g(x)|/f(x))=\n',maxg)
print('max(|f(x)-h(x)|/f(x))=\n',maxh)
print("\nh(x) is more accurate!")
plt.show()


# ## Question 2
# ### Error bounds with Lagrange polynomials.

# #### a
# Let $f(x) = e^{-3x} + e^{2x}$
# . Write a program to calculate and plot the Lagrange polynomial $p_{n−1}(x)$ of $f(x)$ at the Chebyshev points $x_k = \cos((2j - 1)\pi/2n)$ for $j = 1, \ldots, n$. For $n = 4$, over the range $[-1, 1]$, plot $f(x)$ and Lagrange polynomial $p_3(x)$.

# In[37]:


# a
n=4
x=np.linspace(-1,1,101)

# define f(x) as described
def f(x):
    return np.exp(-3*x)+np.exp(2*x)

# creating the list of Chebyshev interpolation points
xk=[]
for j in range(1,n+1):
    xk.append(math.cos((2*j-1)*math.pi/(2*n)))

# construct the new polynomial using Lagrange interpolation as before
# this time, the coefficients aren't necessary, so I just have the vector of y-values
pn=[]
for i in x:
    p=0
    for k in xk:
        temp = [x for x in xk if x != k]
        L=1
        for j in temp:
            L=L*(i-j)/(k-j)
        p+=f(k)*L    
    pn.append(p)
    
# plot, they're pretty similar
print("-"*40)
print("QUESTION 2: Error bounds with Lagrange polynomials.")
print("-"*40)
print("PART A: plot  f(x)  and Lagrange polynomial  p_3(x)")
plt.plot(x,f(x),label='f(x)')
plt.plot(x,pn,label='p_n(x)')
plt.legend()
plt.savefig('2a.png')
plt.show()


# #### b
# Recall from the lectures that the infinity norm for a function $g$ on $[-1, 1]$ is defined as $||g||_\infty = \max_{x\in[-1,1]}|g(x)|$. Calculate $|| f - p_3||_\infty$ by sampling the function at 1,000 equally-spaced points over $[-1, 1]$.

# In[39]:


# b
x = np.linspace(-1,1,1000)
fx = f(x)

# re-doing the above with more points for accuracy
pn=[]
for i in x:
    p=0
    for k in xk:
        temp = [x for x in xk if x != k]
        L=1
        for j in temp:
            L=L*(i-j)/(k-j)
        p+=f(k)*L    
    pn.append(p)
    
# finding the infinity norm of the residual
orig=np.max(abs(fx-pn))
print("PART B: Calculate  ||f−p3||∞  by sampling the function at 1,000 equally-spaced points over  [−1,1] ")
print('\ninfinity norm=\n',orig)


# #### d
# Find a cubic polynomal $p^\dagger_3$ such that $|| f - p^\dagger_3||_\infty < || f - p_3||_\infty$

# In[45]:


# d
x = np.linspace(-1,1,1000)
fx = f(x)
n=4
print("PART D: Find a cubic polynomal  p†3  such that  ||f−p†3||∞<||f−p3||∞")
while True:
    xk=[]
    for j in range(1,n+1):
        xk.append(math.cos((2*j-1)*math.pi/(2*n)))
    xk[1]=xk[1]-np.random.rand()
    xk[2]=xk[2]+np.random.rand()
    
    pm=[]
    for i in x:
        p=0
        for k in xk:
            temp = [x for x in xk if x != k]
            L=1
            for j in temp:
                L=L*(i-j)/(k-j)
            p+=f(k)*L    
        pm.append(p)
    
    new=np.max(abs(fx-pm))
    if new < orig:
        print('\nimproved infinity norm=\n',new)
        print('improved interpolation points=\n',xk)
        break


# ## Question 4: 
# ### Periodic cubic splines

# #### a
# Consider the four points $(t, x) = (0, 0),(1, 1),(2, 0),(3, -1)$. Construct a cubic spline $s_x(t)$ that is piecewise cubic in the four intervals $[0, 1),[1, 2), [2, 3), [3, 4)$. At $t = 0, 1, 2, 3$ the cubics should match the control points, giving eight constraints. At $t = 0, 1, 2, 3$ the first and second derivatives should match, giving and additional eight constraints and allowing $s_x(t)$ to be uniquely determined.
# 
# #### b
# Plot $s_x(t)$ and $\sin(t\pi/2)$ on the interval $[0, 4)$ and show that they are similar

# In[110]:


# defining the basis functions
def c0(x):
    return (x**2)*(3-2*x)
def c1(x):
    return -(x**2)*(1-x)
def c2(x):
    return (x-1)**2*x
def c3(x):
    return 2*x**3-3*x**2+1

# defining the intervals [0,4)
x1=np.linspace(0,1,100001)
x2=np.linspace(1,2,100001)
x3=np.linspace(2,3,100001)
x4=np.linspace(3,4,100001)
x=np.linspace(0,4,400004)

# a/b
# plugging in the coefficients found algebraically
y1a=c0(x1)-0*c1(x1)+1.5*c2(x1)
y2a=-1.5*c1(x2-1)-0*c2(x2-1)+c3(x2-1)
y3a=-c0(x3-2)-0*c1(x3-2)-1.5*c2(x3-2)
y4a=1.5*c1(x4-3)-0*c2(x4-3)-c3(x4-3)
ya=np.sin(x*np.pi/2)

print("-"*40)
print("QUESTION 4: Periodic cubic splines.")
print("-"*40)
print("PART B: Plot  sx(t)  and  sin(tπ/2)  on the interval  [0,4)  and show that they are similar")
# plot for visualization
plt.plot(x1,y1a,label='s(t)',color='r')
plt.plot(x2,y2a,color='r')
plt.plot(x3,y3a,color='r')
plt.plot(x4,y4a,color='r')
plt.plot(x,ya,label='sin(t)')
plt.legend()
plt.savefig('4b_sin.png')
plt.show()


# #### c
# Construct a second cubic spline $s_y(t)$ that goes through the four points $(0, 1), (1, 0), (2, -1), (3, 0)$. Plot $s_y(t)$ and $\cos(t\pi/2)$ on the interval $0 \le t < 4$ and show that they are similar.

# In[52]:


# c
# repeating the above procedure
yb=[math.cos(i*math.pi/2) for i in x]
for i in range(101):
    y1b=-1.5*c1(x1)+c3(x1)
    y2b=-c0(x1)-1.5*c2(x1)
    y3b=1.5*c1(x1)-c3(x1)
    y4b=c0(x1)+1.5*c2(x4-3)

print("PART C: Plot  sy(t)  and  cos(tπ/2)  on the interval  0≤t<4  and show that they are similar")
plt.plot(x1,y1b,label='s(t)',color='r')
plt.plot(x2,y2b,color='r')
plt.plot(x3,y3b,color='r')
plt.plot(x4,y4b,color='r')
plt.plot(x,yb,label='cos(t)')
plt.legend()
plt.savefig("4c_cos.png")
plt.show()


# #### d
# In the $xy$-plane, plot the parametric curve $(s_x(t),s_y(t))$ for $t \in [0, 4)$. Calculate the area enclosed by the parametric curve, and use it to estimate $\pi$ to at least five decimal places, using the relationship $A = \pi r^2$ where $r$ is taken to be $1$.

# In[56]:


# plotting the parametric
plt.plot(y1a,y1b,y2a,y2b,y3a,y3b,y4a,y4b,color='r')
ax=plt.axes()
ax.set_aspect('equal')
plt.savefig("4d.png")
plt.show()

# approximating the area for each region using the triangle formula
area=0
for i in range(len(y1a)-1):
    area += abs(y1a[i]*y1b[i+1]-y1a[i+1]*y1b[i])/2
for i in range(len(y2a)-1):
    area += abs(y2a[i]*y2b[i+1]-y2a[i+1]*y2b[i])/2
for i in range(len(y3a)-1):
    area += abs(y3a[i]*y3b[i+1]-y3a[i+1]*y3b[i])/2
for i in range(len(y4a)-1):
    area += abs(y4a[i]*y4b[i+1]-y4a[i+1]*y4b[i])/2
    
print("PART D: Calculate the area enclosed by the parametric curve, and use it to estimate  π  to at least five decimal places\n")
print('area=\n',area,'\nrelative error from pi=\n',abs(area-math.pi)/math.pi)


# ## Question 5
# ### Image reconstruction using low light.
# Each pixel in the image can be represented as a three-component vector $\mathbf{p} = (R, G, B)$ for the red, green, and blue components. Let $\mathbf{p}^A_k$ be the $k$th pixel of the regular photo, and let $\mathbf{p}^B_k$, $\mathbf{p}^C_k$, and $\mathbf{p}^D_k$
# be the $k$th pixel of the three low-light photos. Here, $k$ is
# indexed from $0$ up to $MN − 1$.

# #### a
# Consider reconstructing the regular photo from the three low-light photos. The
# regular photo pixel could be obtained from the low-light photo pixels using
# $$\mathbf{p}^A_k = F^B\mathbf{p}^B_k+F^C\mathbf{p}^C_k+F^D\mathbf{p}^D_k + \mathbf{p}^{const}$$
# where $F^B, F^C$, and $F^D$ are $3\times3$ matrices and $\mathbf{p}^{const}$ is a vector. Write a program to find the least-squares fit for the 30 components of the matrices $F^B, F^C, F^D$, and the vector $\mathbf{p}^{const}$. Specifically, your program should minimize 
# $$S = \frac{1}{MN}\sum_{k=0}^{MN-1}||F^B\mathbf{p}^B_k + F^C\mathbf{p}^C_k + F^D\mathbf{p}^D_k + \mathbf{p}^{const} - \mathbf{p}^A_k||^2_2.$$
# Calculate $S$ for the fitted values of $F^B, F^C, F^D$, and $\mathbf{p}^{const}$ . Reconstruct a regular image using the pixel values given by Eq. 3 and include it in your writeup. Compare it to the original regular image.

# In[111]:


#reading in the images
low1=io.imread('am205_hw1_files/problem5/objects/400x300/low1.png').astype(np.float64);low1*=1/255.0
low2=io.imread('am205_hw1_files/problem5/objects/400x300/low2.png').astype(np.float64);low2*=1/255.0
low3=io.imread('am205_hw1_files/problem5/objects/400x300/low3.png').astype(np.float64);low3*=1/255.0
regA =io.imread('am205_hw1_files/problem5/objects/400x300/regular.png').astype(np.float64);regA*=1/255.0
b=[low1,low2,low3]

(N,M,z)=regA.shape

# organizing the channels and the A matrix
# A_{k,:}=[low1[R,G,B],low2[R,G,B],low3[R,G,B],1] for 10 elements corresponding to each
#      row of Fi and the constant for the channel
A=np.empty([N*M,10])
y=np.empty([N*M,3])

for image in range(len(b)):
    for color in range(3):
        A[:,3*image+color]=b[image][:,:,color].reshape(M*N)
    
for color in range(3):
    y[:,color] = regA[:,:,color].reshape(M*N)

# use python routine to find the corresponding lambda values for each channel
F=np.linalg.lstsq(A,y)[0]

print("-"*40)
print("QUESTION 5: Image reconstruction using low light.")
print("-"*40)
print("PART A: find the least-squares fit for the 30 components of the matrices  FB,FC,FD , and the vector  pconst\n")
print("F=")
print(F)

reconA=np.empty([N,M,z])
for color in range(3):
    reconA[:,:,color]=np.dot(A,F[:,color]).reshape(N,M)
        
print("\nImage reconstruction")
io.imsave("reconA.png",reconA)
plt.imshow(reconA)


# In[93]:


# finding the residual
S=(np.linalg.norm(regA-reconA))**2/(M*N)
print("PART A: Calculate  S  for the fitted values of  FB,FC,FD , and  pconst\n")        
print('S=\n',S)


# Chris also took a similar set of photos of two of his favorite bears, located in
# the directory problem5/bears. Using the three low-light images of the bears,
# plus your fitted model from part (a), create a reconstructed regular image and
# include it in your writeup. In addition, calculate
# $$T =\frac{1}{MN}\sum^{MN-1}_{k=0}||\mathbf{p}^A_k  p^{A_*}_k||^2_2$$
# where $\mathbf{p}^A_k$ and $\mathbf{p}^{A_*}_k$ are the pixel colors in the original and reconstructed regular images, respectively.

# In[96]:


# same procedure as a with new pictures
low1=io.imread('am205_hw1_files/problem5/bears/400x300/low1.png').astype(np.float64);low1*=1/255.0
low2=io.imread('am205_hw1_files/problem5/bears/400x300/low2.png').astype(np.float64);low2*=1/255.0
low3=io.imread('am205_hw1_files/problem5/bears/400x300/low3.png').astype(np.float64);low3*=1/255.0
regB =io.imread('am205_hw1_files/problem5/bears/400x300/regular.png').astype(np.float64);regB*=1/255.0

b=[low1,low2,low3]

(N,M,z)=regB.shape

# organizing the channels and the A matrix
# A_{k,:}=[low1[R,G,B],low2[R,G,B],low3[R,G,B],1] for 10 elements corresponding to each
#      row of Fi and the constant for the channel
A=np.empty([N*M,10])
y=np.empty([N*M,3])

for image in range(len(b)):
    for color in range(3):
        A[:,3*image+color]=b[image][:,:,color].reshape(M*N)
    
for color in range(3):
    y[:,color] = regB[:,:,color].reshape(M*N)

# use python routine to find the corresponding lambda values for each channel
F=np.linalg.lstsq(A,y)[0]
print("PART B: find the least-squares fit for the 30 components of the matrices  FB,FC,FD , and the vector  pconst\n")
print("F=")
print(F)

reconB=np.empty([N,M,z])
for color in range(3):
    reconB[:,:,color]=np.dot(A,F[:,color]).reshape(N,M)
        
print("\nImage reconstruction")
io.imsave("reconB.png",reconB)
plt.imshow(reconB)


# In[86]:


# same procedure as before
T=1/(N*M)*np.linalg.norm(reconB-regB)**2
print("PART A: Calculate  T  for the fitted values of  FB,FC,FD , and  pconst\n")        
print('T=\n',T)


# ## Question 6: 
# ### Determining hidden chemical sources.
# Suppose that $\rho(x, t)$ represents the concentration of a chemical diffusing in two-dimensional space, where $\mathbf{x} = (x, y)$. The concentration satisfies the diffusion equation $$\frac{\partial\rho}{\partial t}= b\nabla^2\rho = b\bigg(\frac{\partial^2\rho}{\partial x^2} +\frac{\partial^2\rho}{\partial y^2}\cdot\bigg),$$
# where $b$ is the diffusion coefficient. If a localized point source of chemical is introduced at the origin at $t = 0$, its concentration satisfies
# $$\rho_c(\mathbf{x}, t) = \frac{1}{4\pi bt}\exp\bigg(-\frac{x^2 + y^2}{4bt}\bigg).$$

# #### b.
# Set $b = 1$, and now suppose that 49 point sources of chemicals are introduced at
# $t = 0 $with different strengths, on a $7 \times 7$ regular lattice covering the coordinates $x = −3, −2, \ldots , 3$ and $y = −3, −2, \ldots, 3$. By linearity the concentration will satisfy
# $$\rho(\mathbf{x}, t) =\sum^{48}_{k=0}\lambda_k\rho_c(\mathbf{x} - \mathbf{v}_k, t),$$
# where $\mathbf{v}_k$ is the $k$th lattice site and $\lambda_k$ is the strength of the chemical introduced at that site. In the program files, you will find a file measured-concs.txt that contains many measurements of the concentration field, $\rho_i = \rho(\mathbf{x}_i, 4)$, for when $t = 4$. By any means necessary, determine the concentration strengths $\lambda_k$.

# In[112]:


# reading the file and formatting the data
with open('am205_hw1_files/problem6/measured-concs.txt') as f:
    rho_data=f.readlines()
size = len(rho_data)
coords = np.empty([size,2])
conc = np.empty([size,1])
i=0
for line in rho_data:
    coords[i]=line.split()[0],line.split()[1]
    conc[i]=line.split()[-1]
    i+=1

# defining the pc function as described
def pc(x,t):
    return (1/(4*math.pi*t))*np.exp(-(x[0]**2+x[1]**2)/(4*t))

# b

# creating A matrix 
# A_{i,k}=pc(x-vk, t)
A=np.empty([size,49])
for x in range(len(coords)):
    for i in range(7):
        for j in range(7):
            A[x][i+7*j]=pc(coords[x]-np.array([-3+i,-3+j]),4)
            
# using python routine to find lambda strength values
lam=np.linalg.lstsq(A,conc)[0]
lam=lam.reshape(7,7)

print("-"*40)
print("QUESTION 6: Determining hidden chemical sources.")
print("-"*40)
print("PART B: determine the concentration strengths λk\n")
print("λk = ")
print(lam)


# #### c.
# Suppose that the measurements have some experimental error, so that the
# measured values $\tilde\rho_i$ in the file are related to the true values $\rho_i$ according to 
# $$\tilde\rho_i = \rho_i + e_i$$
# where the $e_i$ are normally distributed with mean $0$ and variance $10^{−8}$. Construct
# a hypothetical sample of the true $\rho_i$, and repeat your procedure from part (b) to
# determine the concentrations $\lambda_k$. Repeat this sampling procedure for at least
# 100 times, and use it to measure the standard deviation in the $\lambda_k$ at the lattice
# sites $(0, 0), (1, 0), (2, 0), (3, 0)$. Which of these has the largest standard deviation
# and why?

# In[102]:


# c

# for n=200 iterations, redo the least squares analysis on the hypothetical true concentrations
# and append relevant values (0,0) (1,0) (2,0) (3,0) to a list
n=200
lamb=np.zeros([4,n])
size=conc.shape[0]

for k in range(n):
    p_true = conc-np.random.normal(0.0,1e-4,conc.shape)
    A=np.empty([size,49])
    for x in range(len(coords)):
        for i in range(7):
            for j in range(7):
                A[x][i+7*j]=pc(coords[x]-np.array([-3+i,-3+j]),4)

    lab=np.linalg.lstsq(A,p_true)[0]
    lamb[0][k]+=lab[24]
    lamb[1][k]+=lab[25]
    lamb[2][k]+=lab[26]
    lamb[3][k]+=lab[27]
    
# calculate the standard deviation across the points
print("PART C: measure the standard deviation in the  λk  at the lattice sites  (0,0),(1,0),(2,0),(3,0)\n")
print('stddev=\n',np.nanstd(lamb,axis=1))

