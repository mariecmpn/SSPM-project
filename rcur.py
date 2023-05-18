# RANDOMIZED CUR

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg.interpolative import interp_decomp
from scipy.misc import face

#Example matrix
X = face(gray=True)
X = X.astype(float)
plt.imshow(X)

def rcur(A,k,p=0,q=0):
    m,n = A.shape
    J,X = interp_decomp(A,k)
    Js = J[0:k]
    Z = np.hstack([np.eye(k),X])
    C = A[:,Js]
    _,S,P = sp.linalg.qr(C.T,pivoting = True)
    I = P[0:k]
    R = A[I,:]
    U = Z@np.linalg.pinv(R)
    return C,U,R
  
  # Test
  
  E = [50,100,200,300]
  Err = []
for e in E:
    C,U,R = rcur(X,k,0,0)
    X_hat = C@U@R
    X_hat = X_hat.astype('int8')
    image_stream = Image.fromarray(X_hat)
    plt.imshow(image_stream)
    plt.show()
    err = np.linalg.norm(X-X_hat)/np.linalg.norm(X)
    Err.append(err)
    
plt.plot(E,Err)
plt.scatter(E,Err,marker = '*')
plt.xlabel('k')
plt.title("Relative error $\|X-\hat{X}\|/\|X\|$")

# Singular values of C and R

from sklearn.utils.extmath import randomized_svd

S1,V1,D1 = randomized_svd(C,768,random_state = None)
S2,V2,D2 = randomized_svd(R,50,random_state = None)

x = np.linspace(1,50,50)

plt.plot(x,V1, label = "singular values of C")
plt.plot(x,V2, label = "singular values of R")
plt.yscale('log')
plt.legend()
