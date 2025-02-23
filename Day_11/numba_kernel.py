import numpy as np
from numba import cuda, float32

TPB = 32

@cuda.jit
def matmul_numba(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    tmp = float32(0.0)
    for i in range(bpg):
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]
        
        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]
        
        cuda.syncthreads()

    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

def run_numba(A, B):
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)
    
    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = (A.shape[1] + TPB - 1) // TPB
    blocks_per_grid_y = (A.shape[0] + TPB - 1) // TPB
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    
    start.record()
    matmul_numba[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    end.record()
    end.synchronize()
    
    elapsed_time = cuda.event_elapsed_time(start, end)
    C_host = d_C.copy_to_host()
    return elapsed_time, C_host
