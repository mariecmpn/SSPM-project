import numpy as np 
import math as m 
import scipy.linalg as sp
import matplotlib.pyplot as plt 
from scipy.linalg.interpolative import interp_decomp

################################### RANGE FINDER ALGO ##############################

#fonction that allow to do the ortho basis of Y in stage 1 
def ortho(A):
    """ This function return the orthonormal basis of the column of the matrix A"""
    Q, _ = np.linalg.qr(A)
    return Q 

# Fonction that put a matrix to a power q 
def power_it(A, q):
    n,m = A.shape
    return np.linalg.matrix_power(A@A.T, q)@A

#fonction for algorithm 3 Adaptive randomized Range Finder 
def adaptive_finder(A, e, r=10):
    #we draw the gaussian vectors 
    m,n = np.shape(A)
    O = np.random.randn(n,r)
    #we form the gaussian vectors yi
    Y = A@O
    I= np.eye(m)
    j=0
    #Empty matrix 
    Q=np.zeros((m,1))
    norm_Y = np.linalg.norm(Y, axis=0)# norme of each column 
    while np.max(norm_Y[j:j+r+1])> e*np.sqrt(np.pi/2)/10: 
        j = j+1
        Y[:,j] = (I-Q@ Q.T)@Y[:,j]
        qj= Y[:,j]/np.linalg.norm(Y[:,j])
        Q= np.append(Q,np.resize(qj, (m,1)), axis=1)
        y_jr = (I-Q@Q.T)@ A@ np.random.randn(n,1)
        Y = np.append(Y, np.resize(y_jr, (m,1)), axis=1)
        for i in range(j+1,j+r):
            Y[:,i] = Y[:,i]-qj*(sum(qj*Y[:,i]))
        norm_Y = np.linalg.norm(Y, axis=0)
    return Q

# Algorithm subspace iteration 
def sub_iter(A, rank, q, oversampling = None): #matrix (m*n)
    m,n= np.shape(A)
    if oversampling is None:
        sampling = rank  
    else: 
        sampling = rank+oversampling
    #gaussian matrix 
    O = np.random.randn(n,sampling)
    #we form Y0
    Y0 = A@O
    Q = ortho(Y0)
    for j in range(q):
        ytilde = A.T@Q # Tilde(Y_j)
        Qtilde = ortho(ytilde)
        Y = A@Qtilde
        Q = ortho(Y)
    return Q #Q_q

#fonction that would implement the Algorithm of the stage 2 in the paper it is section 4  
def range_finder(A, rank, q=None, oversampling = None):
    """This function compute the algorithm 2 (Stage 2), with the diferent implementation"""
    m,n = np.shape(A)
    
    if oversampling is None:
        sampling = rank  
    else: 
        sampling = rank+oversampling
        
    #first step draw the gaussian vectors
    Omega = np.random.randn(n, sampling)
    
    #we form the product A*omega 
    if q is None: #basic scheme algo  2
        Y = A @ Omega
    else: #power iteration 
        Y =  power_it(A,q)@Omega #Algorithm 4: Power-Iteration
        
    #construction of Q with QR factorisation 
    Q = ortho(Y)
    
    return Q 

################################### RANDOMIZED SVD ##################################

#alternative for row extraction 
def range_extra(A, rank, oversampling=None):
    m,n = np.shape(A)
    
    if oversampling is None:
        sampling = rank  
    else: 
        sampling = rank+oversampling
    #first step draw the gaussian vectors
    Omega = np.random.randn(n, sampling)
    Y = A @ Omega 
    return Y

#algorithm SVD via row extraction 
def row_extra(A, Q, e):
    #we compute th ID
    k, indx, P= interp_decomp(Q.T, e, rand=True) #compute on Q.T in order to have selection for rows.
    X = np.vstack([np.eye(k), P.T]) # reconstruction of X
    #extraction de de A(J,:)
    Aj = A[indx[0:k], :] 
    Rt, Wt = np.linalg.qr(Aj)  
    Z= X @ Rt
    U, S, Vtilde = np.linalg.svd(Z)
    V = Wt.T@Vtilde
    return U, S, V.T #this return XA(J,:) the ID approx of A


#fonction that would do the algorithm of the reduction of svd based on the range finder algo 
def rsvd(A, rank,q=None, e=None,oversampling=None, return_range=False, subiter=False):
    m, n = A.shape
    
    #Setting the matrix Q with the 3 differents algo for range finder 
    if subiter:
        Q = sub_iter(A,rank, q, oversampling)
    else:
        Q = range_finder(A, rank, q,oversampling) #compute simple range-finder or power-it depending on the presence of q 
        
    #computing the rsvd    
    if e is None:#simple scheme 
        
        #Construct the matrix B 
        B = np.transpose(Q)@A
        #Compute the svd 
        Utilde, S, Vt = np.linalg.svd(B)
        #construc U 
        U = Q @ Utilde
    else: 
        U, S, Vt = row_extra(A,Q, e)
    
    if return_range:
        return U[:,:rank], S[:rank], Vt[:rank, :], Q
    else: 
        return U[:,:rank], S[:rank], Vt[:rank, :]
