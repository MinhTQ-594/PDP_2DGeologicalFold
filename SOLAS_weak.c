#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 15
#define NY 15
#define INF 1e2
#define F 1.0
#define BLOCK_SIZE 4
#define MAX_BLOCKS_X ((NX + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define MAX_BLOCKS_Y ((NY + BLOCK_SIZE - 1) / BLOCK_SIZE)

int noC = 14336; // Number of subdomains the platform can compute simultaneously (for CUDA later)
int MinAct = 0;

// Subdomain structure with ghost cells
typedef struct {
    double data[BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // includes ghost layer
    int CL; // 1 = Open, 0 = Locked
} Subdomain;

Subdomain blocks[MAX_BLOCKS_X][MAX_BLOCKS_Y];
double SD[MAX_BLOCKS_X][MAX_BLOCKS_Y];

double ComputeSD(Subdomain* sd) {
    double sum = 0.0;
    for (int i = 1; i <= BLOCK_SIZE; i++) {
        for (int j = 1; j <= BLOCK_SIZE; j++) {
            sum += sd->data[i][j];
        }
    }
    return sum / (BLOCK_SIZE * BLOCK_SIZE);
}

void InitializeProblem() {
    for (int bx = 0; bx < MAX_BLOCKS_X; bx++) {
        for (int by = 0; by < MAX_BLOCKS_Y; by++) {
            Subdomain* sd = &blocks[bx][by];
            for (int i = 0; i < BLOCK_SIZE + 2; i++) {
                for (int j = 0; j < BLOCK_SIZE + 2; j++) {
                    sd->data[i][j] = INF;
                }
            }
            sd->CL = 0;
            SD[bx][by] = INF;
        }
    }

    int cx = NX / 2;
    int cy = NY / 2;
    int bx = cx / BLOCK_SIZE;
    int by = cy / BLOCK_SIZE;
    int lx = (cx % BLOCK_SIZE) + 1;
    int ly = (cy % BLOCK_SIZE) + 1;
    blocks[bx][by].data[lx][ly] = 0.0;
    blocks[bx][by].CL = 1;
    SD[bx][by] = ComputeSD(&blocks[bx][by]);
}


/// compute su
void ComputeSubdomain(Subdomain* sd, int bx, int by) {
    for (int i = 1; i <= BLOCK_SIZE; i++) {
        for (int j = 1; j <= BLOCK_SIZE; j++) {
            double tx = fmin(sd->data[i-1][j], sd->data[i+1][j]);
            double ty = fmin(sd->data[i][j-1], sd->data[i][j+1]);
            double diff = fabs(tx - ty);

            double updated;
            if (diff >= 1.0 / F)
                updated = fmin(tx, ty) + 1.0 / F;
            else
                updated = (tx + ty + sqrt(2.0 / (F*F) - diff * diff)) / 2.0;

            if (updated < sd->data[i][j])
                sd->data[i][j] = updated;
        }
    }
    SD[bx][by] = ComputeSD(sd);
}

void ActivateNeighbor(int bx, int by, int* CL) {
    if (bx >= 0 && bx < MAX_BLOCKS_X && by >= 0 && by < MAX_BLOCKS_Y) {
        int idx = bx * MAX_BLOCKS_Y + by;
        if (CL[idx] == 0)
            CL[idx] = 1;
    }
}

int BuildScheduleSOLAS(int* L, int* CL) {
    static double oldAv = INF;
    int noActive = 0;
    double sumSD = 0.0;
    double cutT = INF;
    int sxsy = MAX_BLOCKS_X * MAX_BLOCKS_Y;

    for (int bx = 0; bx < MAX_BLOCKS_X; bx++) {
        for (int by = 0; by < MAX_BLOCKS_Y; by++) {
            int idx = bx * MAX_BLOCKS_Y + by;
            if (CL[idx] == 1) {
                noActive++;
                sumSD += SD[bx][by];
            }
        }
    }

    double Av = INF;
    if (noActive > MinAct) {
        Av = sumSD / noActive;
        cutT = Av + 0.4 * fmax(0.0, Av - oldAv);
        oldAv = Av;
    }

    int noSched = 0;
    for (int bx = 0; bx < MAX_BLOCKS_X; bx++) {
        for (int by = 0; by < MAX_BLOCKS_Y; by++) {
            int idx = bx * MAX_BLOCKS_Y + by;
            if (CL[idx] == 1 && SD[bx][by] <= cutT) {
                L[noSched++] = idx;
            }
            MinAct = fmax(2 * noC, (2.0 / 3.0) * fmax(sxsy, noSched));
        }
    }

    return noSched;
}

int ComputeSchedule(int* Schedule, int noSched, int* CL) {
    int noActive = 0;
    for (int i = 0; i < noSched; i++) {
        int id = Schedule[i];
        int bx = id / MAX_BLOCKS_Y;
        int by = id % MAX_BLOCKS_Y;

        if (CL[id] == 1) {
            ComputeSubdomain(&blocks[bx][by], bx, by);
            CL[id] = 0;
            noActive++;
        }
    }
    return noActive;
}

void SyncFromSchedule(int* Schedule, int noSched, int* CL) {
    for (int dir = -1; dir <= 1; dir += 2) {
        for (int i = 0; i < noSched; i++) {
            int id = Schedule[i];
            int bx = id / MAX_BLOCKS_Y;
            int by = id % MAX_BLOCKS_Y;

            // X-direction neighbor
            int nbx = bx + dir;
            if (nbx >= 0 && nbx < MAX_BLOCKS_X) {
                int changed = 0;
                for (int j = 1; j <= BLOCK_SIZE; j++) {
                    double val = blocks[bx][by].data[(dir == -1 ? 1 : BLOCK_SIZE)][j];
                    int ghost_row = (dir == -1 ? BLOCK_SIZE + 1 : 0);
                    if (val < blocks[nbx][by].data[ghost_row][j]) {
                        blocks[nbx][by].data[ghost_row][j] = val;
                        changed = 1;
                    }
                }
                if (changed) ActivateNeighbor(nbx, by, CL);
            }

            // Y-direction neighbor
            int nby = by + dir;
            if (nby >= 0 && nby < MAX_BLOCKS_Y) {
                int changed = 0;
                for (int j = 1; j <= BLOCK_SIZE; j++) {
                    double val = blocks[bx][by].data[j][(dir == -1 ? 1 : BLOCK_SIZE)];
                    int ghost_col = (dir == -1 ? BLOCK_SIZE + 1 : 0);
                    if (val < blocks[bx][nby].data[j][ghost_col]) {
                        blocks[bx][nby].data[j][ghost_col] = val;
                        changed = 1;
                    }
                }
                if (changed) ActivateNeighbor(bx, nby, CL);
            }
        }
    }
}

void PrintGlobalGrid() {
    for (int gx = 0; gx < NX; gx++) {
        for (int gy = 0; gy < NY; gy++) {
            int bx = gx / BLOCK_SIZE;
            int by = gy / BLOCK_SIZE;
            int lx = (gx % BLOCK_SIZE) + 1;
            int ly = (gy % BLOCK_SIZE) + 1;

            if (bx >= MAX_BLOCKS_X || by >= MAX_BLOCKS_Y) {
                printf("   ???  ");
                continue;
            }

            printf("%7.2f ", blocks[bx][by].data[lx][ly]);
        }
        printf("\n");
    }
}

void SolverSOLAS() {
    int CL[MAX_BLOCKS_X * MAX_BLOCKS_Y] = {0};
    int Schedule[MAX_BLOCKS_X * MAX_BLOCKS_Y] = {0};

    for (int bx = 0; bx < MAX_BLOCKS_X; bx++) {
        for (int by = 0; by < MAX_BLOCKS_Y; by++) {
            if (blocks[bx][by].CL == 1) {
                CL[bx * MAX_BLOCKS_Y + by] = 1;
                SD[bx][by] = ComputeSD(&blocks[bx][by]);
            }
        }
    }

    int noSched = BuildScheduleSOLAS(Schedule, CL);
    int noActive = noSched;

    int iter = 0;

    while (noSched > 0 && iter < 100) {
        while ((double)noActive / noSched > 1.0 / 64.0) {
            noActive = ComputeSchedule(Schedule, noSched, CL);
            SyncFromSchedule(Schedule, noSched, CL);
            if (noActive < noC) noActive = 0;
        }
        noSched = BuildScheduleSOLAS(Schedule, CL);
        noActive = noSched;
        iter++;
        if (iter == 11) {
            int a;
            a = 1;
        }
        printf("iter = %d \n", iter);
        PrintGlobalGrid();

    }
}

int main() {
    InitializeProblem();
    SolverSOLAS();
    printf("Final travel time grid (SOLAS):\n");
    PrintGlobalGrid();
    return 0;
}
