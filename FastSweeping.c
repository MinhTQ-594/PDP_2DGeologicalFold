#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <malloc.h>

#define Nx 10
#define Ny 10
#define INF 1e9
#define MAX_ITER 1
#define TOL 1e-4
#define F 1.0 // Hệ số vận tốc sóng không đổi
#define dx 1.0 // Khoảng cách lưới theo chiều x
#define dy 1.0 // Khoảng cách lưới theo chiều y

// Mảng thời gian đến
double T[Nx][Ny];

// Mảng cờ khởi tạo
int isSource[Nx][Ny];

// Tính nghiệm cập nhật tại 1 điểm
double update(int i, int j) {
    double tx = INF, ty = INF;

    if (i > 0) tx = fmin(tx, T[i - 1][j]);
    if (i < Nx - 1) tx = fmin(tx, T[i + 1][j]);

    if (j > 0) ty = fmin(ty, T[i][j - 1]);
    if (j < Ny - 1) ty = fmin(ty, T[i][j + 1]);

    // Giải nghiệm bậc hai (upwind)
    double a = fmin(tx, ty);
    double b = fmax(tx, ty);
    double diff = fabs(tx - ty);

    if (diff >= 1.0 / F) {
        return a + 1.0 / F;
    } else {
        double temp = (tx + ty + sqrt((2.0*dx*dx) / (F * F) - diff * diff)) / 2.0;
        return temp;
    }
}

// Fast Sweeping theo 4 hướng
int fast_sweeping() {
    int iter = 0;
    while (iter++ < MAX_ITER) {
        double max_diff = 0;

        int dirs[4][2] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}; //, {-1, 1}, {-1, -1}, {1, -1}

        for (int d = 0; d < 4; d++) {
            int di = dirs[d][0];
            int dj = dirs[d][1];

            for (int i = (di == 1 ? 0 : Nx - 1); i >= 0 && i < Nx; i += di) {
                for (int j = (dj == 1 ? 0 : Ny - 1); j >= 0 && j < Ny; j += dj) {
                    if (isSource[i][j]) continue;

                    double oldT = T[i][j];
                    double newT = update(i, j);
                    if (newT < oldT) {
                        T[i][j] = newT;
                        double diff = fabs(oldT - newT);
                        if (diff > max_diff) max_diff = diff;
                    }
                }
            }
        }

        if (max_diff < TOL) break;
    }
    return iter;
}

// Thiết lập nguồn ban đầu
void initialize_sources() {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            T[i][j] = INF;
            isSource[i][j] = 0;
        }
    }

    // Ví dụ: nguồn tại tâm
    int cx = Nx / 2, cy = Ny / 2;
    T[cx][cy] = 0;
    isSource[cx][cy] = 1;
}

void print_T() {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            printf("%8.2f ", T[i][j]);
        }
        printf("\n");
    }
}

int main() {
    initialize_sources();
    int iter = fast_sweeping();
    printf("Number of iterations: %d\n", iter);
    printf("Time to reach each point:\n");
    print_T();
    return 0;
}
