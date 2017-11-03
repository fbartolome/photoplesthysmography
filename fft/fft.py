import numpy as np
import time


def fft(x, type):
    start_time = time.time()
    if type == 'iterative':
        ret = fft_iterative(x)
        print("Iterative time ", time.time() - start_time)
    elif type == 'recursive':
        ret = fft_recursive(x)
        print("Recursive time ", time.time() - start_time)
    elif type == "dft":
        ret = dft_classic(x)
        print("Dft time", time.time() - start_time)
    else:
        raise Exception("The chosen FFT algorithm type is not a valid one")
    return ret


def fft_recursive_impl(x):
    N = x.shape[0]

    if N == 1:
        return x
    else:
        even = fft_recursive_impl(x[::2])
        odd = fft_recursive_impl(x[1::2])
        #Complex coefficients for butterfly calculation
        W = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.concatenate([even + W[: (int(N/2))] * odd,
                               even + W[(int(N/2)) :] * odd])


def fft_recursive(x):

    if np.log2(x.shape[0])%1>0:
        raise ValueError("size of x must be a power of 2")

    return np.around(fft_recursive_impl(x), 8)


def fft_shift(x):
    for i in range(int(len(x)/2)):
        aux = x[i]
        x[i] = x[int(len(x)/2) + i]
        x[int(len(x)/2) + i]= aux

    return x


def fft_iterative(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    X = np.zeros(N, dtype= np.complex)
    W = np.zeros(N, dtype=np.complex)

    #If not a power of 2, abort
    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")
    log2 = int(np.log2(N))

    for i in range(0,N):
        #index bit reversal
        X[int('{:0{width}b}'.format(i,width=log2)[::-1], 2)] = x[i]
        #Calculation of complex coefficients
        W[i] = np.exp(-2j * np.pi * i / N)

    butterfly_len = 1

    #Butterfly composition from each sub parts
    for butterfly_index in range(1,log2+1):

        half_len = butterfly_len
        butterfly_len *= 2
        #Coefficient jump for current butterflies
        w_jump = 2**(log2-butterfly_index)
        #Number of butterflies
        block_limit = int(N/butterfly_len)

        #For each butterfly
        for block_count in range(0,block_limit):

            #For the first half of elements of it
            for i in range(0,half_len):

                #Select the first element of the current butterfly
                first_butterfly_element = X[ block_count*butterfly_len + i]
                #Select de complimentary element which will be added
                second_butterfly_element = X[ block_count*butterfly_len + i + half_len]

                #Calculate the addition of both with the corresponding complex coefficient Wi
                X[block_count * butterfly_len + i] = first_butterfly_element + W[w_jump * i] * second_butterfly_element
                #Calculate the second addition using them, whith the other corresponding coefficient Wj
                X[block_count * butterfly_len + i + half_len] = first_butterfly_element + W[w_jump * (
                i + half_len)] * second_butterfly_element

    #returns fft
    return np.around(X,8)


def dft_classic(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
