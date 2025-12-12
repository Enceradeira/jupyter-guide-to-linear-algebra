from . import laguide as lag
import numpy as np
import pdb

# Solves for X in UX=B


def BackSubstitution(U, B):
    m = U.shape[0]  # number of rows to build X, e.g 3
    X = np.zeros((m, 1))

    for i in range(m-1, -1, -1):  # e.g i=(2,1,0)
        X[i] = B[i]           # e.g i=(2,1,0)
        for j in range(i+1, m):  # e.g j(i=2)=(), j(i=1)=(2), j(i=0)=(1,2)
            X[i] -= U[i][j]*X[j]  # subtract already found X-value from equation
        if (U[i][i] != 0):     # pivot != 0
            X[i] /= U[i][i]    # divide by pivot
        else:
            raise ValueError(f"Zero entry found in U pivot position {i}.")
    return X


def RowReduction(A):
    # =============================================================================
    # A is a NumPy array that represents an augmented matrix of dimension n x (n+1)
    # associated with a linear system.  RowReduction returns B, a NumPy array that
    # represents the row echelon form of A.  RowReduction may not return correct
    # results if the the matrix A does not have a pivot in each column.
    # =============================================================================

    m = A.shape[0]  # A has m rows
    n = A.shape[1]  # It is assumed that A has m+1 columns

    B = np.copy(A).astype('float64')

    # For each step of elimination, we find a suitable pivot, move it into
    # position and create zeros for all entries below.
    for k in range(m):
        # Set pivot as (k,k) entry
        pivot = B[k][k]
        pivot_row = k

        # Find a suitable pivot if the (k,k) entry is zero
        while (pivot == 0 and pivot_row < m-1):
            pivot_row += 1
            pivot = B[pivot_row][k]

        # Swap row if needed
        if (pivot_row != k):
            print(f"row swap of row {k} and {pivot_row}")
            B = lag.RowSwap(B, k, pivot_row)

        # If pivot is nonzero, carry on with elimination in column k
        if (pivot != 0):
            B = lag.RowScale(B, k, 1./B[k][k])
            for i in range(k+1, m):
                B = lag.RowAdd(B, k, i, -B[i][k])
        else:
            print("Pivot could not be found in column", k, ".")

    return B


def SolveSystem(A, B):
    # =============================================================================
    # A is a NumPy array that represents a matrix of dimension n x n.
    # B is a NumPy array that represents a matrix of dimension n x 1.
    # SolveSystem returns a NumPy array of dimension n x 1 such that AX = B.
    # If the system AX = B does not have a unique solution, SolveSystem may not
    # generate correct results.
    # =============================================================================

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("SolveSystem accepts only square arrays.")
        return
    n = A.shape[0]  # n is number of rows and columns in A

    # 1. Join A and B to make the augmented matrix
    A_augmented = np.hstack((A, B))

    # 2. Carry out elimination
    R = RowReduction(A_augmented)

    # 3. Split R back to nxn piece and nx1 piece
    B_reduced = R[:, n:n+1]
    A_reduced = R[:, 0:n]

    # 4. Do back substitution
    X = BackSubstitution(A_reduced, B_reduced)
    return X


def RREF(A):
    S = RowReduction(A)
    S1 = lag.RowAdd(S, 2, 1, -S[1, 2])
    S2 = lag.RowAdd(S1, 1, 0, -S1[0, 1])
    S3 = lag.RowAdd(S2, 2, 0, -S2[0, 2])
    return S3


def Inverse(A):
    # =============================================================================
    # A is a NumPy array that represents a matrix of dimension n x n.
    # Inverse computes the inverse matrix by solving AX=I where I is the identity.
    # If A is not invertible, Inverse will not return correct results.
    # =============================================================================

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Inverse accepts only square arrays.")
        return
    n = A.shape[0]  # n is number of rows and columns in A

    I = np.eye(n)

    # The augmented matrix is A together with all the columns of I.  RowReduction is
    # carried out simultaneously for all n systems.
    A_augmented = np.hstack((A, I))
    R = RowReduction(A_augmented)

    Inverse = np.zeros((n, n))

    # Now BackSubstitution is carried out for each column and the result is stored
    # in the corresponding column of Inverse.
    A_reduced = R[:, 0:n]
    for i in range(0, n):
        B_reduced = R[:, n+i:n+i+1]
        Inverse[:, i:i+1] = BackSubstitution(A_reduced, B_reduced)

    return (Inverse)
