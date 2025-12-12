import numpy as np
from scipy.linalg import inv


def mod_inverse(a, m):
    """Calculate modular multiplicative inverse of a modulo m"""
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None


def modular_inverse_matrix(A, mod):
    """Calculate modular inverse of matrix A modulo mod"""
    # Calculate determinant
    det = int(np.round(np.linalg.det(A)))

    # Find modular inverse of determinant
    det_inv = mod_inverse(det % mod, mod)

    if det_inv is None:
        raise ValueError("Matrix is not invertible modulo {}".format(mod))

    # Calculate adjugate matrix (inverse * determinant)
    A_inv = inv(A)
    adjugate = np.round(A_inv * det).astype(int)

    # Multiply by inverse of determinant modulo mod
    A_inv_mod = (det_inv * adjugate) % mod

    return A_inv_mod


def convert_to_numbers(text, alphabet):
    number_message = []
    for char in text:
        number_message.append(alphabet.index(char))
    return number_message


def convert_to_chunks(number_message, alphabet):
    remainder = len(number_message) % 4
    if remainder != 0:
        # Pad with zeros to make it divisible by 4
        padding = 4 - remainder
        pad_number = alphabet.index(' ')
        number_message = np.append(number_message, [pad_number] * padding)
    chunks = int(len(number_message) / 4)
    P = number_message.reshape((chunks, 4))
    return P.transpose()


def nr_to_alphabet(C, alphabet):
    encrypted_message = ''
    for j in range(C.shape[1]):
        for i in range(C.shape[0]):
            encrypted_message += alphabet[C[i, j] % len(alphabet)]
    return encrypted_message


def alphabet_to_nr(encrypted_message, alphabet):
    number_of_columns = int(len(encrypted_message) / 4)
    decryptionC = np.zeros((4, number_of_columns), dtype=int)
    k = 0
    for j in range(number_of_columns):
        for i in range(4):
            decryptionC[i, j] = alphabet.index(encrypted_message[k])
            k += 1
    return decryptionC


# ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET = [' ', '.', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z']

B = np.array([[1, 0, -2, -1],
              [3, -1, -3, 2],
              [2, 0, -4, 4],
              [2, 1, -1, -1]])
B_inv = modular_inverse_matrix(B, len(ALPHABET))

plaintext = "WE DONT HAVE NUMBERS IN OUR ALPHABET. WE HAVE TO SPELL TWO."

number_message = convert_to_numbers(plaintext, ALPHABET)
array_message = np.array(number_message)
P = convert_to_chunks(array_message, ALPHABET)

C = B@P

encrypted_message = nr_to_alphabet(C, ALPHABET)
print("Encrypted message:", encrypted_message)

decryptionC = alphabet_to_nr(encrypted_message, ALPHABET)

decryptionP = B_inv@decryptionC

decrypted_text = nr_to_alphabet(decryptionP, ALPHABET)
print("Decrypted message:", decrypted_text)
