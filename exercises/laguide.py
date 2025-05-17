
import numpy as np

def RowSwap(A,k,l):
     n = A.shape[1]
     B = np.copy(A).astype('float64')

     for j in range(n):
         temp = B[k][j]
         B[k][j] = B[l][j]
         B[l][j] = temp
     return B

def RowScale(A,k,scale):
    n = A.shape[1]
    B = np.copy(A).astype('float64')

    for j in range(n):
        B[k][j] *= scale

    return B

    

def RowAdd(A,k,l,scale):
    n = A.shape[1]
    B = np.copy(A).astype('float64')

    for j in range(n):
        B[l][j] += B[k][j] * scale

    return B
