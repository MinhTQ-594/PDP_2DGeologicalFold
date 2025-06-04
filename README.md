# **PDP_2DGeologicalFold**

## LAS.c
The sequential code

## LAS_CUDA.cu
The parallel code with CUDA - initial with active source block only

## LAS_CUDA_1.cu
The parallel code with CUDA - initial with all active block

## SOLAS_weak.c
Sequential implementation of SOLAS without PMM support for the Compute Subdomain.

## SOLAS.c
Sequential implementation of SOLAS with PMM support for the Compute Subdomain.

## SOLAS_level1.cu
CUDA-based parallel implementation of SOLAS using a single level of parallelism. The full implementation described in the paper includes two levels of parallelism, but that was too complex for me to implement.

## Program
run LAS.exe to see the sequential result

run LAS_CUDA.exe to see the CUDA result

run LAS_CUDA_1.exe to see the other CUDA result version
