import numpy as np
import ctypes
from numba_kernel import run_numba

N = 4096
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Run Numba kernel
time_numba, C_numba = run_numba(A, B)
print(f"Numba execution time: {time_numba:.3f} ms")

# Run CUDA kernel
cuda_lib = ctypes.CDLL("./cuda_kernel.so")
cuda_lib.run_cuda.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)
]

time_cuda = ctypes.c_float()
C_cuda = np.zeros((N, N), dtype=np.float32)

cuda_lib.run_cuda(A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   C_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   ctypes.c_int(N),
                   ctypes.byref(time_cuda))

print(f"CUDA execution time: {time_cuda.value:.3f} ms")

# Validate results
np.testing.assert_allclose(C_numba, C_cuda, rtol=1e-3)
print("Results match!")
