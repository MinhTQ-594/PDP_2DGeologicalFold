#include <stdio.h>
#include <math.h>

#define NX 10
#define NY 10
#define B 5 // Kích thước của mỗi subdomain - Blocksize = 5*5
#define SDX (NX / B) // Số lượng subdomains theo chiều ngang
#define SDY (NY / B) // Số lượng subdomains theo chiều dọc
#define INF 1e9
#define MAX_ITER 1000
#define TOL 1e-4
#define F 1.0

__global__ double update(double T[NX][NY], int i, int j) {
    double tx = INF, ty = INF;

    if (i > 0) tx = fmin(tx, T[i - 1][j]);
    if (i < NX - 1) tx = fmin(tx, T[i + 1][j]);
    if (j > 0) ty = fmin(ty, T[i][j - 1]);
    if (j < NY - 1) ty = fmin(ty, T[i][j + 1]);

    double diff = fabs(tx - ty);
    if (diff >= 1.0 / F) return fmin(tx, ty) + 1.0 / F;
    else return (tx + ty + sqrt(2.0 / (F * F) - diff * diff)) / 2.0;
}

// Khởi tạo: subdomain chứa nguồn
__global__ void init_source(double T[NX][NY], int isSource[NX][NY], int* active) {
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                T[i][j] = INF;
                isSource[i][j] = 0;
                active[(i / B) * SDY + (j / B)] = 1;
            }
    }
}

__global__ void compute_subdomains(double T[NX][NY], int isSource[NX][NY], int* active, int* new_active, int* any_change) {
    int sx = blockIdx.x;
    int sy = threadIdx.x;
    int idx = sx * SDY + sy;
    if (!active[idx]) return;

    int changed = 0;
    int i0 = sx * B;
    int j0 = sy * B;

    for (int i = i0; i < i0 + B && i < NX; i++) {
        for (int j = j0; j < j0 + B && j < NY; j++) {
            if (isSource[i][j]) continue;

            double oldT = T[i][j];
            double newT = update(T, i, j);
            if (newT < oldT) {
                T[i][j] = newT;
                changed = 1;
            }
        }
    }

    if (changed) {
        *any_change = 1;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                int nsx = sx + dx;
                int nsy = sy + dy;
                if (nsx >= 0 && nsx < SDX && nsy >= 0 && nsy < SDY)
                    new_active[nsx * SDY + nsy] = 1;
            }
    }
}

void print_T(double T[NX][NY]) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            printf("%8.2f ", T[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // Host -> Device
    //1a. Delare and Allocate Mem on CPU
    double cpu_T[NX][NY];
    int cpu_isSource[NX][NY] = {0};
    //1b. Delare and Allocate Mem on GPU
    double gpu_T[NX][NY];
    int gpu_isSource[NX][NY];
    int *gpu_active, *gpu_new_active, *gpu_any_change;
    cudaMalloc((void**)&gpu_T, sizeof(cpu_T));
    cudaMalloc((void**)&gpu_isSource, sizeof(cpu_isSource));
    cudaMalloc((void**)&gpu_active, sizeof(int) * SDX * SDY); // Biến để lưu trạng thái của các subdomains
    cudaMalloc((void**)&gpu_new_active, sizeof(int) * SDX * SDY);  // Biến để lưu trạng thái của các subdomains sau mỗi lần cập nhật, biến này sẽ được gán cho gpu_active sau mỗi lần cập nhật
    cudaMalloc((void**)&gpu_any_change, sizeof(int)); // Biến để kiểm tra xem có thay đổi nào xảy ra trong quá trình tính toán
    
    //2. Copy Input from CPU to GPU
    cudaMemcpy(gpu_T, cpu_T, sizeof(cpu_T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_isSource, cpu_isSource, sizeof(cpu_isSource), cudaMemcpyHostToDevice);
    cudaMemset(gpu_active, 0, sizeof(int) * SDX * SDY);

    //3. Define Block and Thread Structure
    dim3 dimGrid(SDX, SDY); // Mỗi subdomain là một block
    dim3 dimBlock(B, B); // Mỗi subdomain chỉ có một thread
    // Khởi tạo nguồn
    // 4. Call Kernel
    init_source<<<1, 1, sizeof(int)>>>(gpu_T, gpu_isSource, gpu_active);

    int iter = 0;
    while (iter++ < MAX_ITER) {
        cudaMemset(gpu_any_change, 0, sizeof(int));
        cudaMemset(gpu_new_active, 0, sizeof(int) * SDX * SDY);

        compute_subdomains<<<dimGrid, dimBlock, B*sizeof(double)>>>(gpu_T, gpu_isSource, gpu_active, gpu_new_active, gpu_any_change);

        // Kiểm tra kết thúc
        int flag = 0;
        cudaMemcpy(&flag, gpu_any_change, sizeof(int), cudaMemcpyDeviceToHost);
        if (!flag) break;

        cudaMemcpy(gpu_active, gpu_new_active, sizeof(int) * SDX * SDY, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(cpu_T, gpu_T, sizeof(cpu_T), cudaMemcpyDeviceToHost);

    print_T(cpu_T);
    printf("Final matrix:\n");
    printf("Iterations: %d\n", iter);

    // Cleanup
    cudaFree(gpu_T); cudaFree(gpu_isSource); cudaFree(gpu_active);
    cudaFree(gpu_new_active); cudaFree(gpu_any_change);
    free(cpu_T); free(cpu_isSource);
    cudaDeviceReset(); // Reset the device to clean up resources

    return 0;
}
