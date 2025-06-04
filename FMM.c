#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 5                // Kích thước lưới
#define INF 1e10            // Giá trị vô cực
#define F 1.0               // Vận tốc sóng
#define dx (5.0 / (N))  // Khoảng cách lưới
#define epsilon 1e-6        // Ngưỡng hội tụ
#define SOURCE_I (N / 2)
#define SOURCE_J (N / 2)

// Khởi tạo ma trận T với giá trị INF, đặt nguồn tại trung tâm
double** initialize() {
    double **T = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        T[i] = (double*)malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) {
            T[i][j] = INF;
        }
    }
    T[SOURCE_I][SOURCE_J] = 0.0;
    return T;
}

// Cập nhật giá trị T theo phương pháp upwind và giải phương trình Eikonal
void update_T(double **T_new, double **T_old) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == SOURCE_I && j == SOURCE_J) {
                T_new[i][j] = 0.0;
                continue;
            }

            double a = INF, b = INF;

            // Chiều x
            if (i > 0) a = fmin(a, T_old[i - 1][j]);
            if (i < N - 1) a = fmin(a, T_old[i + 1][j]);

            // Chiều y
            if (j > 0) b = fmin(b, T_old[i][j - 1]);
            if (j < N - 1) b = fmin(b, T_old[i][j + 1]);

            double temp;
            if (fabs(a - b) >= F * dx) {
                temp = fmin(a, b) + F * dx;
            } else {
                double sum = a + b;
                double disc = sum * sum - 2.0 * (a * a + b * b - F * F * dx * dx);
                if (disc >= 0) {
                    temp = 0.5 * (sum + sqrt(disc));
                } else {
                    temp = INF;
                }
            }

            T_new[i][j] = fmin(T_old[i][j], temp);
        }
    }
}

// Tính sai số lớn nhất giữa 2 bước lặp
double max_diff(double **T_new, double **T_old) {
    double max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == SOURCE_I && j == SOURCE_J) continue; // Bỏ qua điểm nguồn
            if (T_new[i][j] < INF && T_old[i][j] < INF) {
                double diff = fabs(T_new[i][j] - T_old[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
    }
    return max_diff;
}

// In ma trận
void print_matrix(double **T) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (T[i][j] >= INF - 1)
                printf(" INF     ");
            else
                printf("%.2f    ", T[i][j]);
        }
        printf("\n");
    }
}

// Giải chính
int main() {
    double diff = INF;
    double **T_old = initialize();
    double **T_new = initialize();

    int max_iter = 1000;
    int iter = 0;
    while (iter < max_iter) {
        update_T(T_new, T_old);
        // diff = max_diff(T_new, T_old);
        iter++;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                T_old[i][j] = T_new[i][j]; // Cập nhật giá trị cũ
            }
        }
    }

    printf("Number of iterations: %d\n", iter);
    printf("Resulting T matrix:\n");
    print_matrix(T_new);

    // Giải phóng bộ nhớ
    for (int i = 0; i < N; i++) {
        free(T_old[i]);
        free(T_new[i]);
    }
    free(T_old);
    free(T_new);

    return 0;
}
