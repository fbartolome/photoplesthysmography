import numpy as np

def FFT_recursive(x):

    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N == 1:
        return x

    if N%2 > 0:
        raise ValueError('x should be a power of 2')

    else:
        X_even = FFT_recursive(x[::2])
        X_odd = FFT_recursive(x[1::2])
        # TODO factor puede estar precalculado en un mapa
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[: (int(N/2))] * X_odd,
                               X_even + factor[(int(N/2)) :] * X_odd])

def FFT(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :(int(X.shape[1] / 2))]
        X_odd = X[:, (int(X.shape[1] / 2)):]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])
    return X.ravel()


def FFT_shift(x):
    for i in range(int(len(x)/2)):
        aux = x[i]
        x[i] = x[int(len(x)/2) + i]
        x[int(len(x)/2) + i]= aux

    return x
