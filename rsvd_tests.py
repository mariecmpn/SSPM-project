import fonctions.rsvd as rd #see file rsvd.py

import numpy as np 
import matplotlib.pyplot as plt 
from pylab import *
import scipy as sp 

# Artificial matrix

#write a matrix that have rapid decaying singular value 
def decaying_rapidly(m,n, q):
    U =rd.ortho( np.random.normal(0,1, (m,n)))
    V = rd.ortho( np.random.normal(0,1, (m,n)))
    S = [q** i for i in range(n)]
    return U @np.diag(S)@V.T
  
  #creation of artificial matrix 
def slow_matrix(m,n, q,a=0.3):
    U =rd.ortho( np.random.normal(0,1, (m,n)))
    V = rd.ortho( np.random.normal(0,1, (m,n)))
    S = [a+i*q if i<n//2 else 0.7 for i in range(1,n+1)]
    return U@np.diag(S)@V.T
  
  # Applications
  
  # Matrix with singular values that decay rapidly
  
#we set a matrix A 
m,n = 600, 600
A = decaying_rapidly(m,n,0.6)
#and form its svd 
_, As, _= np.linalg.svd(A)

#visualisation of the spectrum
fig, ax = plt.subplots(1,1)
ax.scatter(range(100), np.log10(As[:100]), color = '#11accd', s=2, label = r'$\log_{10}(\sigma_{\ell+1})$',
           marker = 'o')
ax.plot(range(100), np.log10(As[:100]), color='#11accd', linewidth=1)
ax.set_title('Matrix with rapid decayin values ')
plt.legend()
plt.show()

#Computing the rsvd on A 
l = range(1, 100, 5)# rank
min_ = list()
errors = list()

for k in l:
    Q = rd.range_finder(A, k)
    err = np.linalg.norm((np.eye(m)- Q @ np.transpose(Q))@ A) # frobinus norm  
    errors.append(np.log10(err))
    min_.append(np.log10(As[k+1]))
    
 #generating figures:
fig, ax = plt.subplots(1, 1)
ax.scatter(l, min_, color='#11accd', s=30,
               label=r'$\log_{10}(\sigma_{\ell+1})$', marker='v')
ax.scatter(l, errors, color='#807504', s=30, label=r'$\log_{10}(e_{\ell})$',
               marker='o')
ax.plot(l, min_, color='#11accd', linewidth=1)
ax.plot(l, errors, color='#807504', linewidth=1)

ax.set_ylabel(r'$Log_{10}$ of errors ')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title('Exponentially decaying singular values')

plt.legend()
plt.tight_layout()
plt.show()

# Matrix with singular values that decay slowly

m,n = 200, 200
B = slow_matrix(m,n, 0.2,a=0.3 ) 
_, Bs, _= np.linalg.svd(B)

fig, ax = plt.subplots(1,1)
ax.scatter(range(100), np.log10(Bs[:100]), color = '#11accd', s=3, 
           label = r'$\log_{10}(\sigma_{\ell+1})$', marker = 'o')
ax.plot(range(100), np.log10(Bs[:100]), color='#11accd', linewidth=1)
ax.set_title('Matrix with slow decaying singular values ')
plt.legend()
plt.show()

#Computing the rsvd on A 
l = range(1, 100, 5)# rank
min_ = list()
errors = list()

for k in l:
    Q = rd.range_finder(B, k)
    err = np.linalg.norm((np.eye(m)- Q @ np.transpose(Q))@ B) #norm de frobinus 
    errors.append(np.log10(err))
    min_.append(np.log10(Bs[k+1]))
    
 fig, ax = plt.subplots(1, 1)
ax.scatter(l, min_, color='#11accd', s=30,
               label=r'$\log_{10}(\sigma_{\ell+1})$', marker='v')
ax.scatter(l, errors, color='#807504', s=30, label=r'$\log_{10}(e_{\ell})$',
               marker='o')
ax.plot(l, min_, color='#11accd', linewidth=1)
ax.plot(l, errors, color='#807504', linewidth=1)

ax.set_ylabel(r'$Log_{10}$ of errors ')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title('Slowly decaying singular values')

plt.legend()
plt.tight_layout()
plt.show()

# Visualisation of the power iteration with the SVD

#First we normalise the singular values of B:
S_true = [s / Bs.max() for s in Bs]
x = range(len(S_true)) #counting each singular values 

#settinfg the plot
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(11, 5)
ax.scatter(x[:125], S_true[:125], color='purple', s=1)
ax.plot(x[:125], S_true[:125], label='True singular values', color='purple', marker='o', linewidth=1)

#Computing the Power-Iteration range finder algorithm with different values of power q:
qs = [(1, '#11accd', '*'), (2, '#807504', 'v'), (3, '#bc2612', 'd')]
for q, color, marker in qs:
    B_new = rd.power_it(B,q)
    S = np.linalg.svd(B_new, compute_uv=False)
    S_new = [s / S.max() for s in S]
    ax.scatter(x[:125], S_new[:125], color=color, s=1)
    ax.plot(x[:125], S_new[:125], label=r'$q = %s$' % q, color=color, marker=marker)
ax.set_title('Normalized singular values with $q$ power iterations')
ax.set_ylabel('Normalized magnitude')
plt.legend()
plt.tight_layout()
plt.show()

# Computing RSVD with power iteration

mins = list()
q2list = {
    0: ([], '#11accd', 'o'),
    1: ([], '#807504', 'v'),
    2: ([], '#bc2612', 'd'),
    3: ([], '#236040', '*'), 
    4: ([],'#552360', 'p' )
    }

l = [20, 40, 60, 80, 100]
qs = [0, 1, 2, 3, 4]


#Computing the rsvd with power iteration on B
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=Bs[k+1]
    mins.append(np.log10(a))
    #approximation for each q 
    for q in qs: 
        Q = rd.range_finder(B, k, q) #range finder with power iteration scheme 
        err = np.linalg.norm((np.eye(m)- Q @ np.transpose(Q))@ B) # approximation error  
        q2list[q][0].append(np.log10(err))
        
#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)

#plot for approximation 
for q in qs:
    data, color, marker = q2list[q]
    ax.scatter(l, data, s=30, label=r'$q = %s$' % q, marker=marker,
                   color=color)
    ax.plot(l, data, linewidth=1, color=color)

ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with power iteration ')
plt.legend()
plt.tight_layout()
plt.show()

m,n = 200, 200
B = slow_matrix(m,n, 0.2,a=0.3 ) 
_, Bs, _= np.linalg.svd(B)


mins = list()

# different exponent q 
q2list = {
    0: ([], '#11accd', 'o'),
    1: ([], '#807504', 'v'),
    2: ([], '#bc2612', 'd'),
    3: ([], '#236040', '*'), 
    4: ([],'#552360', 'p' )
    }
#ranks 
l = [20, 40 ,60, 80, 100]
qs= [0,1,2,3,4]
fig, ax = plt.subplots(1,1)
fig.set_size_inches(8,5)

#theorical minimum 
 
for k in l:
    mins.append(np.log10(Bs[k+1]))
    for q in qs:
        Q = rd.sub_iter(B, k, q)
        e = np.linalg.norm(((np.eye(m)- Q @ np.transpose(Q)) @ B), 'fro')
        q2list[q][0].append(np.log10(e))

#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)

for q in qs:
    error, color, mark = q2list[q]
    ax.scatter(l, error, s=30, label=r'$q = %s$' % q, marker=mark,
                   color=color)
    ax.plot(l, error, color=color)

ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with subiteration ')
plt.legend()
plt.tight_layout()
plt.show()

# Computing the RSVD with subspace iteration

from scipy.misc import face 

C=face(gray=True) #generate the image of the racoon (dense matrix)
C = C.astype(float)
_,Cs,_ = np.linalg.svd(C.T)
m,n=np.shape(C)
m,n

#The image 
plt.imshow(C, cmap='gray')

#rsvd on the second dimension n=1024
rank = [20,50, 100, 300]
for elt in rank:
    U,S,Vt = rd.rsvd(C.T, elt)
    Cr = (U*S)@Vt
    plt.imshow(Cr.T, cmap='gray')
    plt.show()
    
mins= []
err = []

l = [20, 50, 80, 100, 300]
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=Cs[k+1]
    mins.append(np.log10(a))
    Q = rd.range_finder(C.T, k)
    e = np.linalg.norm((np.eye(n)- Q@ Q.T)@ C.T)
    err.append(np.log10(e))

ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)
ax.scatter(l, err, color= '#11accd', s=30, label= r'$\log_{10}(e_{\ell})$')
ax.plot(l, err, color= '#11accd')
ax.set_ylabel(r'$Log_{10}$ of errors ')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title('Slowly decaying singular values')

plt.legend()
plt.tight_layout()
plt.show()

mins = list()
q2list = {
    0: ([], '#11accd', 'o'),
    1: ([], '#807504', 'v'),
    2: ([], '#bc2612', 'd'),
    3: ([], '#236040', '*'), 
    4: ([],'#552360', 'p' )
    }

l = [20, 50, 80, 100, 300]
qs = [0, 1, 2, 3, 4]


#Computing the rsvd with power iteration on B
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=Cs[k+1]
    mins.append(np.log10(a))
    #approximation for each q 
    for q in qs: 
        Q = rd.sub_iter(C, k, q)
        err = np.linalg.norm((np.eye(m)- Q @ np.transpose(Q))@ C) #norm de frobinus 
        q2list[q][0].append(np.log10(err))
#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)
#plot for approximation 
for q in qs:
    data, color, marker = q2list[q]
    ax.scatter(l, data, s=30, label=r'$q = %s$' % q, marker=marker,
                   color=color)
    ax.plot(l, data, linewidth=1, color=color)

ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with subspace iteration ')
plt.legend()
plt.tight_layout()
plt.show()

# Computing RSVD with subspace iteration and row iteration

mins = list()
m,n = A.shape
err = list()
l = [20, 40, 60, 80, 100]
#Computing the rsvd with power iteration on B
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=As[k+1]
    mins.append(np.log10(a))
    _,_,_, Q = rd.rsvd(A, k,e=0.1, return_range=True )
    e = np.linalg.norm((np.eye(m)- Q @ Q.T)@ A) #norm de frobinus
    err.append(np.log10(e))

#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)
#plot for row extraction 
ax.scatter(l, err, color='#11accd', s=30, label=r'$log_{10}(e_l)$', marker='o')
ax.plot(l, err, color='#11accd', linewidth=1)
ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with row extraction ')
plt.legend()
plt.tight_layout()
plt.show()

mins = list()
m,n = A.shape
q2list = {
    0: ([], '#11accd', 'o'),
    1: ([], '#807504', 'v'),
    2: ([], '#bc2612', 'd'),
    3: ([], '#236040', '*'), 
    4: ([],'#552360', 'p' )
    }

l = [20, 40, 60, 80, 100]
qs = [0, 1, 2, 3, 4]


#Computing the rsvd with power iteration on B
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=As[k+1]
    mins.append(np.log10(a))
    #approximation for each q 
    for q in qs: 
        _,_,_, Q = rd.rsvd(A, k, q,e=0.1, return_range=True,subiter=True )
        err = np.linalg.norm((np.eye(m)- Q @ Q.T)@ A) #norm de frobinus 
        q2list[q][0].append(np.log10(err))
#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)
#plot for approximation 
for q in qs:
    data, color, marker = q2list[q]
    ax.scatter(l, data, s=30, label=r'$q = %s$' % q, marker=marker,
                   color=color)
    ax.plot(l, data, linewidth=1, color=color)

ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with row extraction and subspace iteration ')
plt.legend()
plt.tight_layout()
plt.show()

mins = list()
m,n = C.shape
q2list = {
    0: ([], '#11accd', 'o'),
    1: ([], '#807504', 'v'),
    2: ([], '#bc2612', 'd'),
    3: ([], '#236040', '*'), 
    4: ([],'#552360', 'p' )
    }

l = [20, 40, 60, 80, 100]
qs = [0, 1, 2, 3, 4]


#Computing the rsvd with power iteration on B
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 5)
for k in l:
    #theorical error 
    a=Cs[k+1]
    mins.append(np.log10(a))
    #approximation for each q 
    for q in qs: 
        _,_,_, Q = rd.rsvd(C, k, q, e=0.1, return_range=True,subiter=True )
        err = np.linalg.norm((np.eye(m)- Q @ Q.T)@ C) #norm de frobinus 
        q2list[q][0].append(np.log10(err))
#plot for theorical minimum
ax.scatter(l, mins, color='gray', s=30, label='Minimum error', marker='s')
ax.plot(l, mins, color='gray', linewidth=1)
#plot for approximation 
for q in qs:
    data, color, marker = q2list[q]
    ax.scatter(l, data, s=30, label=r'$q = %s$' % q, marker=marker,
                   color=color)
    ax.plot(l, data, linewidth=1, color=color)

ax.set_ylabel('$log_10$ Magnitude')
ax.set_xlabel(r'Random samples $\ell$')
ax.set_title(r'Approximation error $e_{\ell}$ with row extraction and subspace iteration ')
plt.legend()
plt.tight_layout()
plt.show()

# Adaptative range finder

epsilon= 0.2
s=0
for i in range(100):
    s+=rd.adaptive_finder(A, 0.2).shape[1]
s/100

borne = 0.2/(20/np.pi)
print("error of tolerance for the matrix Y (gaussian matrix) ", borne)
print("The theorical error at rank ", int(s/100), "is", S[int(s/100)])

epsilon = 0.2
s=0
for i in range(100):
    s+=adaptive_finder(B, epsilon).shape[1]
s/100

borne = epsilon/(20/np.pi)
print("error of tolerance for the matrix Y (gaussian matrix) ", borne)
print("The theorical error at rank ", int(s/100)-2, "is", Bs[int(s/100)-2])
