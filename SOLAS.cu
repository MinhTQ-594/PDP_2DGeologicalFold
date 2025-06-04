#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 10
#define NY 10
#define INF 100
#define F 1.0
#define BLOCK_SIZE 4
#define MAX_BLOCKS_X ((NX + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define MAX_BLOCKS_Y ((NY + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define MAX_BLOCKS (MAX_BLOCKS_X * MAX_BLOCKS_Y)
#define THREADS_PER_BLOCK 1024

__device__ __host__ inline int getIndex(int bx, int by) {
    return bx * MAX_BLOCKS_Y + by;
}

struct Subdomain {
    double data[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    int CL;
};

__device__ double computeSDDevice(Subdomain* sd) {
    double sum = 0.0;
    for (int i = 1; i <= BLOCK_SIZE; i++) {
        for (int j = 1; j <= BLOCK_SIZE; j++) {
            sum += sd->data[i][j];
        }
    }
    return sum / (BLOCK_SIZE * BLOCK_SIZE);
}

__device__ void ComputeSubdomain(Subdomain* sd, int bx, int by, double* SD) {
    for (int i = 1; i <= BLOCK_SIZE; i++) {
        for (int j = 1; j <= BLOCK_SIZE; j++) {
            double tx = fmin(sd->data[i - 1][j], sd->data[i + 1][j]);
            double ty = fmin(sd->data[i][j - 1], sd->data[i][j + 1]);
            double diff = fabs(tx - ty);

            double updated;
            if (diff >= 1.0 / F)
                updated = fmin(tx, ty) + 1.0 / F;
            else
                updated = (tx + ty + sqrt(2.0 / (F * F) - diff * diff)) / 2.0;

            if (updated < sd->data[i][j])
                sd->data[i][j] = updated;
        }
    }
    SD[getIndex(bx, by)] = computeSDDevice(sd);
}

__global__ void ComputeScheduleKernel(Subdomain* d_blocks, double* d_SD, int* d_schedule, int* d_CL, int noSched, int* d_noActive) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= noSched) return;

    int id = d_schedule[tid];
    int bx = id / MAX_BLOCKS_Y;
    int by = id % MAX_BLOCKS_Y;

    if (d_CL[id] == 1) {
        ComputeSubdomain(&d_blocks[id], bx, by, d_SD);
        d_CL[id] = 0;
        atomicAdd(d_noActive, 1);
    }
}

__global__ void syncGhostCellsKernel(Subdomain* d_blocks, int* d_schedule, int noSched) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= noSched) return;

    int id = d_schedule[tid];
    int bx = id / MAX_BLOCKS_Y;
    int by = id % MAX_BLOCKS_Y;

    for (int dir = -1; dir <= 1; dir += 2) {
        int nbx = bx + dir;
        if (nbx >= 0 && nbx < MAX_BLOCKS_X) {
            for (int j = 1; j <= BLOCK_SIZE; j++) {
                int idx1 = getIndex(bx, by);
                int idx2 = getIndex(nbx, by);
                d_blocks[idx2].data[(dir == -1 ? BLOCK_SIZE + 1 : 0)][j] =
                    d_blocks[idx1].data[(dir == -1 ? 1 : BLOCK_SIZE)][j];
            }
        }
        int nby = by + dir;
        if (nby >= 0 && nby < MAX_BLOCKS_Y) {
            for (int i = 1; i <= BLOCK_SIZE; i++) {
                int idx1 = getIndex(bx, by);
                int idx2 = getIndex(bx, nby);
                d_blocks[idx2].data[i][(dir == -1 ? BLOCK_SIZE + 1 : 0)] =
                    d_blocks[idx1].data[i][(dir == -1 ? 1 : BLOCK_SIZE)];
            }
        }
    }
}

Subdomain* h_blocks;
double* h_SD;
int h_CL[MAX_BLOCKS];
int h_schedule[MAX_BLOCKS];

double oldAv = INF;
int noC = 1;

void InitializeProblem() {
    h_blocks = (Subdomain*)malloc(MAX_BLOCKS * sizeof(Subdomain));
    h_SD = (double*)malloc(MAX_BLOCKS * sizeof(double));

    for (int bx = 0; bx < MAX_BLOCKS_X; bx++) {
        for (int by = 0; by < MAX_BLOCKS_Y; by++) {
            int id = getIndex(bx, by);
            Subdomain* sd = &h_blocks[id];
            for (int i = 0; i < BLOCK_SIZE + 2; i++) {
                for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                    sd->data[i][j] = INF;
                }
            }
            sd->CL = 0;
            h_SD[id] = INF;
        }
    }

    int cx = NX / 2;
    int cy = NY / 2;
    int bx = cx / BLOCK_SIZE;
    int by = cy / BLOCK_SIZE;
    int lx = (cx % BLOCK_SIZE) + 1;
    int ly = (cy % BLOCK_SIZE) + 1;
    int id = getIndex(bx, by);
    h_blocks[id].data[lx][ly] = 0.0;
    h_blocks[id].CL = 1;
    h_SD[id] = 0.0;
    h_CL[id] = 1;
}

int BuildSchedule(int* schedule) {
    int noActive = 0;
    double sumSD = 0.0;
    double cutT = INF;

    for (int id = 0; id < MAX_BLOCKS; id++) {
        if (h_CL[id] == 1) {
            noActive++;
            sumSD += h_SD[id];
        }
    }

    if (noActive > 2 * noC) {
        double Av = sumSD / noActive;
        cutT = Av + 0.4 * fmax(0.0, Av - oldAv);
        oldAv = Av;
    }

    int noSched = 0;
    for (int id = 0; id < MAX_BLOCKS; id++) {
        if (h_CL[id] == 1 && h_SD[id] < cutT) {
            schedule[noSched++] = id;
        }
    }
    return noSched;
}

void PrintGlobalGrid() {
    for (int gx = 0; gx < NX; gx++) {
        for (int gy = 0; gy < NY; gy++) {
            int bx = gx / BLOCK_SIZE;
            int by = gy / BLOCK_SIZE;
            int lx = (gx % BLOCK_SIZE) + 1;
            int ly = (gy % BLOCK_SIZE) + 1;
            int id = getIndex(bx, by);
            printf("%7.2f ", h_blocks[id].data[lx][ly]);
        }
        printf("\n");
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    noC = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    printf("GPU supports %d active threads across %d SMs. noC set to %d\n", prop.maxThreadsPerMultiProcessor, prop.multiProcessorCount, noC);

    InitializeProblem();

    Subdomain* d_blocks;
    double* d_SD;
    int* d_schedule;
    int* d_CL;
    int* d_noActive;

    cudaMalloc(&d_blocks, MAX_BLOCKS * sizeof(Subdomain));
    cudaMalloc(&d_SD, MAX_BLOCKS * sizeof(double));
    cudaMalloc(&d_schedule, MAX_BLOCKS * sizeof(int));
    cudaMalloc(&d_CL, MAX_BLOCKS * sizeof(int));
    cudaMalloc(&d_noActive, sizeof(int));

    cudaMemcpy(d_blocks, h_blocks, MAX_BLOCKS * sizeof(Subdomain), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SD, h_SD, MAX_BLOCKS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CL, h_CL, MAX_BLOCKS * sizeof(int), cudaMemcpyHostToDevice);

    // time execution cal
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //SolverSOLAS

    int noSched = BuildSchedule(h_schedule);
    int noActive = noSched;

    while (noSched > 0) {
        while ((double)noActive / noSched > 1.0 / 64.0) {
            cudaMemcpy(d_schedule, h_schedule, noSched * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(d_noActive, 0, sizeof(int));

            int blocks = (noSched + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            ComputeScheduleKernel << <blocks, THREADS_PER_BLOCK >> > (d_blocks, d_SD, d_schedule, d_CL, noSched, d_noActive);
            cudaDeviceSynchronize();

            syncGhostCellsKernel << <blocks, THREADS_PER_BLOCK >> > (d_blocks, d_schedule, noSched);
            cudaDeviceSynchronize();

            cudaMemcpy(h_blocks, d_blocks, MAX_BLOCKS * sizeof(Subdomain), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_SD, d_SD, MAX_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_CL, d_CL, MAX_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&noActive, d_noActive, sizeof(int), cudaMemcpyDeviceToHost);

            if (noActive < noC) noActive = 0;
        }
        noSched = BuildSchedule(h_schedule);
        noActive = noSched;
    }
    //
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Execution time: %.2f ms \n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //

    printf("Final travel time grid (CUDA SOLAS):\n");
    PrintGlobalGrid();

    cudaFree(d_blocks);
    cudaFree(d_SD);
    cudaFree(d_schedule);
    cudaFree(d_CL);
    cudaFree(d_noActive);
    free(h_blocks);
    free(h_SD);
    return 0;
}