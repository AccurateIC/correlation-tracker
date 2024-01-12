import numpy as np
def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)

