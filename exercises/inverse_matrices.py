import numpy as np
import laguide as lag

def main():
    # Example code to demonstrate functionality
    matrix = np.array([[1, 2], [3, 4]])
    try:
        inverse_matrix = np.linalg.inv(matrix)
        print("Inverse of the matrix is:")
        print(inverse_matrix)
    except np.linalg.LinAlgError:
        print("Matrix is not invertible.")

if __name__ == "__main__":
    main()
