import tensorly as tl
import numpy as np 
import matplotlib.pyplot as plt 
from tensorly.decomposition import tucker

#Function for TRP: with gaussian; rademacher ; uniform distributions

#First we create a function for DRMS
def DRMs(n,k, law="g"):
    laws ={'g','u', 'r'}
    if law == 'g':
        Omega = np.random.normal(0,1, size= (n,k))
    if law == 'u':
        Omega = np.random.uniform(-1, 1, size = (n,k)) * np.sqrt(3)
    if law ==  'r':
        Omega = np.random.choice([-1,0,1],(n,k), p=[1/6, 2/3, 1/6]) *np.sqrt(3)#sparse more of 0 bcause high probability
    return Omega

#Fonction for the TRP 
def TRP(n_array, k, law="g"):
    T = []
    for n in n_array:
        Omega = DRMs(n,k,law)
        T.append(Omega)
    return tl.tenalg.khatri_rao(T)



#fonction for factors sketching Vn 
def factor_Sketch(X,ks, law="g"):
    factors =[]
    Omegas =[]
    for i,n in enumerate(X.shape): #gives the count and the values 
        n_array = list(X.shape)
        del n_array[i]
        omega = TRP(n_array,ks[i], law) #then the dimension would be (I_(-n), ks[i])
        V = tl.unfold(X,i) @ omega #tl.unfold.shape = (n, I_(-n))
        factors.append(V)
        Omegas.append(omega)
    return factors, Omegas
        
#fonction for the core sketching: 
def core_Sketch(X,s, law="g"):
    Phis = []
    core = X
    sh = list(core.shape)
    for mode in range(len(sh)):
        phi= DRMs(sh[mode], s[mode], law)
        Phis.append(phi)
        core= tl.tenalg.mode_dot(core, phi.T,mode)
    return core, Phis
    
#assembling to compute tucker sketching

def Tucker_Sketch(X,ks, s, law="g"):
    return factor_Sketch(X,ks, law), core_Sketch(X,s,law)


#fonction that allow to do the ortho basis of Y in stage 1 
def ortho(A):
    """ This function return the orthonormal basis of the column of the matrix A"""
    Q, _ = np.linalg.qr(A)
    return Q


# Tensor approximation by Two-pass Sketch 
def Two_Pass(X,k,law="g"):
    Qs = []
    factors,_ = factor_Sketch(X,k, law)
    #orthogonal basis 
    for sketch in factors:
        Q=ortho(np.array(sketch))
        Qs.append(Q)
    #core approximation 
    core_tensor = X
    N = len(factors)
    for mode in range(N):
        Q = Qs[mode]
        core_tensor= tl.tenalg.mode_dot(core_tensor, Q.T, mode) #W 
        
    # the approximation of the tensors X_hat
    X_hat = tl.tucker_to_tensor((core_tensor, Qs))
    return X_hat, core_tensor, Qs

#Tensor approximation by One-Pass
def One_Pass(X,ks,s, law="g"):
    Qs = []
    factors, Omegas = factor_Sketch(X, ks)
    core,Phis= core_Sketch(X,s)
    for factor in factors:
        Q = ortho(factor)
        Qs.append(Q)
    for mode in range(len(factors)):
        core = tl.tenalg.mode_dot(core, np.linalg.pinv(Phis[mode].T@Qs[mode]), mode)
    X_hat = tl.tucker_to_tensor((core, Qs))
    return X_hat, core 


# Tensor error metrics 

def error_eval(X,X_hat):
    err = X-X_hat
    return np.linalg.norm(tl.tensor_to_vec(err)) #euclidean norm of vectorization 

def relative_err(X,X_hat):
    return error_eval(X, X_hat)/ np.linalg.norm(tl.tensor_to_vec(X))

def regret(X,X_hat, k):
    core,factors = tucker(X, k)
    X_HOI = tl.tucker_to_tensor((core, factors))
    return (error_eval(X, X_hat)- error_eval(X, X_HOI))/ np.linalg.norm(tl.tensor_to_vec(X))

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
