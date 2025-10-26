#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <iomanip>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {}
    
    void randomFill() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(1.0, 10.0);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }
    
    const std::vector<double>& operator[](int i) const { return data[i]; }
    std::vector<double>& operator[](int i) { return data[i]; }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
};

Matrix multiplySequential(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }
    
    int m = A.getRows();
    int n = B.getCols();
    int p = A.getCols();
    
    Matrix result(m, n);
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < p; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

Matrix multiplyParallelSimple(const Matrix& A, const Matrix& B, int num_threads) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }
    
    int m = A.getRows();
    int n = B.getCols();
    int p = A.getCols();
    
    Matrix result(m, n);
    
    auto worker = [&](int start_row, int end_row) {
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int k = 0; k < p; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
    };
    
    std::vector<std::thread> threads;
    int rows_per_thread = m / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? m : (t + 1) * rows_per_thread;
        threads.emplace_back(worker, start, end);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return result;
}

void saveResultsToCSV(const std::vector<int>& sizes, 
                     const std::vector<std::vector<double>>& speedups) {
    std::ofstream file("speedup_results.csv");
    file << "MatrixSize,1Thread,2Threads,4Threads,8Threads\n";
    
    for (size_t i = 0; i < sizes.size(); ++i) {
        file << sizes[i] << "x" << sizes[i];
        for (double speedup : speedups[i]) {
            file << "," << std::fixed << std::setprecision(3) << speedup;
        }
        file << "\n";
    }
    file.close();
}

void runComprehensiveAnalysis() {
    std::vector<int> sizes = {100, 200, 300, 500, 800};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::vector<std::vector<double>> all_speedups;
    
    std::cout << "Comprehensive Parallel Performance Analysis\n";
    std::cout << "===========================================\n";
    
    for (int size : sizes) {
        std::cout << "\nAnalyzing " << size << "x" << size << " matrices:\n";
        std::cout << "Threads | Time(s) | Speedup\n";
        
        try {
            Matrix A(size, size);
            Matrix B(size, size);
            A.randomFill();
            B.randomFill();
            
            // Sequential version
            auto start = std::chrono::high_resolution_clock::now();
            Matrix seqResult = multiplySequential(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            double seqTime = std::chrono::duration<double>(end - start).count();
            
            std::vector<double> speedups;
            
            for (int threads : thread_counts) {
                start = std::chrono::high_resolution_clock::now();
                Matrix parResult = multiplyParallelSimple(A, B, threads);
                end = std::chrono::high_resolution_clock::now();
                double parTime = std::chrono::duration<double>(end - start).count();
                
                double speedup = seqTime / parTime;
                speedups.push_back(speedup);
                
                std::cout << std::setw(7) << threads << " | " 
                         << std::setw(7) << std::fixed << std::setprecision(3) << parTime << " | "
                         << std::setw(7) << std::fixed << std::setprecision(3) << speedup << "\n";
            }
            
            all_speedups.push_back(speedups);
            
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }
    
    // Save results
    saveResultsToCSV(sizes, all_speedups);
    std::cout << "\nResults saved to 'speedup_results.csv'\n";
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    runComprehensiveAnalysis();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;
    
    std::cout << "\nTotal execution time: " << total_time.count() << " seconds\n";
    std::cout << "Available hardware threads: " << std::thread::hardware_concurrency() << "\n";
    
    return 0;
}
