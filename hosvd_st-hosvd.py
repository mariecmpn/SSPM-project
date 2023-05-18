# Tucker decomposition

import numpy as np 
import math as m 
import scipy.linalg as sp
import matplotlib.pyplot as plt 
from PIL import Image
import tensorly as tl


############Example of a tensor##############
image = Image.open('path to the image')
X = np.asarray(image)
    
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
