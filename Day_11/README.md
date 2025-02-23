### ** Compile the CUDA Kernel**
```sh
nvcc -O3 -use_fast_math -lineinfo -Xcompiler -fPIC -shared -o cuda_kernel.so matmul_kernel.cu
```

### **Run the Benchmark**
```sh
python benchmark.py
```
This will execute both the Numba and CUDA implementations and compare their execution times.

