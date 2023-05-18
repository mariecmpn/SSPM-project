# Tucker decomposition

import numpy as np 
import math as m 
import scipy.linalg as sp
import matplotlib.pyplot as plt 
from PIL import Image
import tensorly as tl


############Example of a tensor##############
image = Image.open('/Users/mariecompain/Desktop/Cours/M1/SSPM/image.jpg')
X = np.asarray(image)

################RSVD code####################

#fonction that allow to do the ortho basis of Y in stage 1 
def ortho(A):
    """ This function return the orthonormal basis of the column of the matrix A"""
    Q, _ = np.linalg.qr(A)
    return Q 

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

def row_extra(A, Q, e):
    #we compute th ID
    k, indx, P= interp_decomp(Q.T, e, rand=True) #compute on Q.T in order to have selection for rows.
    X = np.vstack([np.eye(k), P.T])[np.argsort(indx),:] # reconstruction of X
    #extraction de de A(J,:)
    Aj = A[indx[1:k+1], :] 
    R, W = np.linalg.qr(Aj)  
    Z= X @ R
    U, S, Vtilde = np.linalg.svd(Z)
    V = W.T@Vtilde
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
        U = Q@Utilde
    else: 
        U, S, Vt = row_extra(A,Q, e)
    
    if return_range:
        return U, S, Vt, Q
    else: 
        return U, S, Vt
    
#####################HOSVD###########################


# test tensor
Xx = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24]]])


from sklearn.utils.extmath import randomized_svd

def hosvd(X,R):
    N = len(X.shape)
    fibers = []
    core = X
    for mode in range(N):
        A,_,_ = randomized_svd(tl.base.unfold(X, mode), R[mode], random_state = None) #random_state = None
        fibers.append(A)
        core = tl.tenalg.mode_dot(core, A.T, mode)
    return core, fibers

I = np.array([20,50,100,200])
E = []

for i in I:
    G, fibers = hosvd(X, [i,i,3])

    X_hat = tl.tucker_to_tensor((G,fibers))
    
    Err = X-X_hat
    m,n,p = Err.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                sum = sum + Err[i][j][k]**2
    err = np.sqrt(sum) / tl.norm(X, order = 2)
    E.append(err)

    X_hat = X_hat.astype('int8')
    image_stream = Image.fromarray(X_hat, 'RGB')
    plt.imshow(image_stream)
    plt.show()

print(E)
plt.plot(I,E)
plt.scatter(I,E,marker = '*')
plt.title('Relative error $\|X-\hat{X} \| / |X\|$')
plt.show()
#print("X = ", X)
#print("X_hat = ", X_hat)


###################ST-HOSVD########################

from scipy.sparse.linalg import svds

#def sthosvd(X,r):
    #N = len(X.shape)
    #G = X
    #fibers = []
    #for mode in range(N):
        #U,S,V = svds(tl.unfold(G,mode),r[mode])
        #G = tl.tenalg.mode_dot(G, np.diag(S)@V, mode)
        #fibers.append(U)
    #return G,fibers


def st_hosvd(tensor, target_ranks):
    """
    Require:
    - tensor X
    - target rank r
    Return: Tucker decomposition of X: G, [A_1,...,A_N]
    """
    original_shape = tensor.shape
    transforming_shape = list(original_shape)
    G = tensor
    arms = []
    for n in range(len(original_shape)):
        G = tl.unfold(G, n)
        U, S, V = randomized_svd(G, target_ranks[n], random_state = None)
        arms.append(U)
        G = np.diag(S) @ V
        transforming_shape[n] = target_ranks[n]
        G = tl.fold(G, n, transforming_shape)
    return G, arms

I = np.array([20,50,100,200])
E = []

for i in I:
    G, fibers = st_hosvd(X, [i,i,3])

    X_hat = tl.tucker_to_tensor((G,fibers))
    
    Err = X-X_hat
    m,n,p = Err.shape
    sum = 0
    #err = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                #err = max(err,abs(Err[i][j][k]))
                sum = sum + Err[i][j][k]**2
    err = np.sqrt(sum) / tl.norm(X, order = 2)
    E.append(err)

    X_hat = X_hat.astype('int8')
    image_stream = Image.fromarray(X_hat, 'RGB')
    plt.imshow(image_stream)
    plt.show()

print(E)
plt.plot(I,E)
plt.scatter(I,E,marker = '*')
plt.title('Relative error $\|X-\hat{X} \| / |X\|$')
plt.show()
