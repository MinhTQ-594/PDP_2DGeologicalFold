#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 10
#define NY 10
#define Blocksize 5
#define SDX (NX / Blocksize)
#define SDY (NY / Blocksize)
#define INF 1e9
#define MAX_ITER 1000
#define F 1.0

#define INDEX(i, j) ((i) * NY + (j))

__device__ double update(double* T, int i, int j) {
    double tx = INF, ty = INF;

    if (i > 0) tx = fmin(tx, T[INDEX(i - 1, j)]);
    if (i < NX - 1) tx = fmin(tx, T[INDEX(i + 1, j)]);
    if (j > 0) ty = fmin(ty, T[INDEX(i, j - 1)]);
    if (j < NY - 1) ty = fmin(ty, T[INDEX(i, j + 1)]);

    double diff = fabs(tx - ty);
    if (diff >= 1.0 / F) return fmin(tx, ty) + 1.0 / F;
    else return (tx + ty + sqrt(2.0 / (F * F) - diff * diff)) / 2.0;
}

__global__ void init_source(double* T, int* isSource, int* active) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < NX && j < NY) {
        if (!isSource[INDEX(i, j)]) // giữ lại điểm nguồn
            T[INDEX(i, j)] = INF;
    }

    int sx = i / Blocksize;
    int sy = j / Blocksize;
    if (sx < SDX && sy < SDY) {
        active[sx * SDY + sy] = 1;
    }
}

__global__ void compute_subdomains(double* T, int* isSource, int* active, int* new_active, int* any_change) {
    int sx = blockIdx.x;
    int sy = threadIdx.x;
    int idx = sx * SDY + sy;
    if (!active[idx]) return;

    int i0 = sx * Blocksize;
    int j0 = sy * Blocksize;
    int changed = 0;

    for (int i = i0; i < i0 + Blocksize && i < NX; i++) {
        for (int j = j0; j < j0 + Blocksize && j < NY; j++) {
            if (isSource[INDEX(i, j)]) continue;

            double oldT = T[INDEX(i, j)];
            double newT = update(T, i, j);
            if (newT < oldT) {
                T[INDEX(i, j)] = newT;
                changed = 1;
            }
        }
    }

    if (changed) {
        *any_change = 1;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nsx = sx + dx;
                int nsy = sy + dy;
                if (nsx >= 0 && nsx < SDX && nsy >= 0 && nsy < SDY)
                    new_active[nsx * SDY + nsy] = 1;
            }
        }
    }
}

void print_T(double* T) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            printf("%8.2f ", T[INDEX(i, j)]);
        }
        printf("\n");
    }
}

int main() {
    double* cpu_T = (double*)malloc(sizeof(double) * NX * NY);
    int* cpu_isSource = (int*)calloc(NX * NY, sizeof(int));

    // Đặt source tại (0, 0)
    int cx = NX / 2;
    int cy = NY / 2;
    cpu_isSource[INDEX(cx, cy)] = 1;
    cpu_T[INDEX(cx, cy)] = 0.0;

    // GPU memory
    double* gpu_T;
    int* gpu_isSource;
    int *gpu_active, *gpu_new_active, *gpu_any_change;

    cudaMalloc(&gpu_T, sizeof(double) * NX * NY);
    cudaMalloc(&gpu_isSource, sizeof(int) * NX * NY);
    cudaMalloc(&gpu_active, sizeof(int) * SDX * SDY);
    cudaMalloc(&gpu_new_active, sizeof(int) * SDX * SDY);
    cudaMalloc(&gpu_any_change, sizeof(int));

    cudaMemcpy(gpu_T, cpu_T, sizeof(double) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_isSource, cpu_isSource, sizeof(int) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemset(gpu_active, 0, sizeof(int) * SDX * SDY);

    dim3 initBlock(NX, NY);
    init_source<<<1, initBlock>>>(gpu_T, gpu_isSource, gpu_active);

    dim3 grid(SDX);
    dim3 block(SDY);

    int iter = 0;
    while (iter++ < MAX_ITER) {
        cudaMemset(gpu_any_change, 0, sizeof(int));
        cudaMemset(gpu_new_active, 0, sizeof(int) * SDX * SDY);

        compute_subdomains<<<grid, block>>>(gpu_T, gpu_isSource, gpu_active, gpu_new_active, gpu_any_change);

        int flag = 0;
        cudaMemcpy(&flag, gpu_any_change, sizeof(int), cudaMemcpyDeviceToHost);
        if (!flag) break;

        cudaMemcpy(gpu_active, gpu_new_active, sizeof(int) * SDX * SDY, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(cpu_T, gpu_T, sizeof(double) * NX * NY, cudaMemcpyDeviceToHost);

    printf("Iterations: %d\n", iter);
    printf("Final travel time field:\n");
    print_T(cpu_T);


    // Cleanup
    cudaFree(gpu_T); cudaFree(gpu_isSource);
    cudaFree(gpu_active); cudaFree(gpu_new_active); cudaFree(gpu_any_change);
    free(cpu_T); free(cpu_isSource);
    cudaDeviceReset();

    return 0;
}