#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NX 10
#define NY 10
#define Blocksize 5
#define INF 1e9
#define MAX_ITER 1000
#define F 1.0

#define INDEX(i, j) ((i) * NY + (j))

double update(double* T, int i, int j) {
    double tx = INF, ty = INF;

    if (i > 0) tx = fmin(tx, T[INDEX(i - 1, j)]);
    if (i < NX - 1) tx = fmin(tx, T[INDEX(i + 1, j)]);
    if (j > 0) ty = fmin(ty, T[INDEX(i, j - 1)]);
    if (j < NY - 1) ty = fmin(ty, T[INDEX(i, j + 1)]);

    double diff = fabs(tx - ty);
    if (diff >= 1.0 / F) return fmin(tx, ty) + 1.0 / F;
    else return (tx + ty + sqrt(2.0 / (F * F) - diff * diff)) / 2.0;
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
    double* T = (double*)malloc(sizeof(double) * NX * NY);
    int* isSource = (int*)calloc(NX * NY, sizeof(int)); // Mảng cờ nguồn khởi tạo tất cả các giá trị = 0 - đánh dấu các điểm nguồn

    // Khởi tạo T và điểm nguồn tại giữa lưới
    for (int i = 0; i < NX * NY; i++) T[i] = INF;
    int cx = NX / 2;
    int cy = NY / 2;
    isSource[INDEX(cx, cy)] = 1;
    T[INDEX(cx, cy)] = 0.0;

    // Khởi tạo mảng active ban đầu
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            if (!isSource[INDEX(i, j)])
                T[INDEX(i, j)] = INF;
        }
    }

    int iter = 0;
    while (iter++ < MAX_ITER) {
        int any_change = 0;

        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                if (isSource[INDEX(i, j)]) continue;
                    double oldT = T[INDEX(i, j)];
                    double newT = update(T, i, j);
                    if (newT < oldT) {
                        T[INDEX(i, j)] = newT;
                        any_change = 1;
                    }
                        }
                    }
        if (!any_change) break;
    }

    printf("Iterations: %d\n", iter);
    printf("Final travel time field:\n");
    print_T(T);

    // Cleanup
    free(T);
    free(isSource);

    return 0;
}
