#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <thread>
#include "main.cpp"

class MatrixTest {
private:
    static const double EPSILON;

public:
    static void runAllTests() {
        std::cout << "Running Matrix Multiplication Tests...\n";
        std::cout << "=====================================\n";
        
        testConstructor();
        testRandomFill();
        testMatrixDimensions();
        testMatrixMultiplicationSmall();
        testMatrixMultiplicationRectangular();
        testThreadBoundaries();
        
        std::cout << "\n All tests passed successfully!\n";
    }

private:
    static void testConstructor() {
        std::cout << "Testing constructor and basic properties... ";
        
        Matrix m(3, 4);
        assert(m.getRows() == 3);
        assert(m.getCols() == 4);
        
        // Проверка инициализации нулями
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                assert(std::abs(m[i][j] - 0.0) < EPSILON);
            }
        }
        
        std::cout << "PASS\n";
    }

    static void testRandomFill() {
        std::cout << "Testing random fill... ";
        
        Matrix m(5, 5);
        m.randomFill();
        
        bool hasNonZero = false;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (m[i][j] != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            if (hasNonZero) break;
        }
        assert(hasNonZero); // Должны быть ненулевые значения
        
        // Проверка диапазона значений (1.0 - 10.0)
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                assert(m[i][j] >= 1.0 && m[i][j] <= 10.0);
            }
        }
        
        std::cout << "PASS\n";
    }

    static void testMatrixDimensions() {
        std::cout << "Testing matrix dimensions validation... ";
        
        Matrix A(2, 3);
        Matrix B(3, 4);
        
        // Это должно работать
        Matrix result = multiplySequential(A, B);
        assert(result.getRows() == 2);
        assert(result.getCols() == 4);
        
        std::cout << "PASS\n";
    }

    static void testMatrixMultiplicationSmall() {
        std::cout << "Testing small matrix multiplication... ";
        
        Matrix A(2, 2);
        Matrix B(2, 2);
        
        // A = [1, 2; 3, 4]
        A[0][0] = 1; A[0][1] = 2;
        A[1][0] = 3; A[1][1] = 4;
        
        // B = [5, 6; 7, 8]
        B[0][0] = 5; B[0][1] = 6;
        B[1][0] = 7; B[1][1] = 8;
        
        // Ожидаемый результат: [19, 22; 43, 50]
        Matrix result = multiplySequential(A, B);
        
        assert(std::abs(result[0][0] - 19) < EPSILON);
        assert(std::abs(result[0][1] - 22) < EPSILON);
        assert(std::abs(result[1][0] - 43) < EPSILON);
        assert(std::abs(result[1][1] - 50) < EPSILON);
        
        std::cout << "PASS\n";
    }

    static void testMatrixMultiplicationRectangular() {
        std::cout << "Testing rectangular matrix multiplication... ";
        
        Matrix A(2, 3);
        Matrix B(3, 4);
        
        // A = [1, 2, 3; 4, 5, 6]
        A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
        A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;
        
        // B = [7, 8, 9, 10; 11, 12, 13, 14; 15, 16, 17, 18]
        B[0][0] = 7;  B[0][1] = 8;  B[0][2] = 9;  B[0][3] = 10;
        B[1][0] = 11; B[1][1] = 12; B[1][2] = 13; B[1][3] = 14;
        B[2][0] = 15; B[2][1] = 16; B[2][2] = 17; B[2][3] = 18;
        
        // Ожидаемый результат: [74, 80, 86, 92; 173, 188, 203, 218]
        Matrix result = multiplySequential(A, B);
        
        assert(std::abs(result[0][0] - 74) < EPSILON);
        assert(std::abs(result[0][1] - 80) < EPSILON);
        assert(std::abs(result[0][2] - 86) < EPSILON);
        assert(std::abs(result[0][3] - 92) < EPSILON);
        assert(std::abs(result[1][0] - 173) < EPSILON);
        assert(std::abs(result[1][1] - 188) < EPSILON);
        assert(std::abs(result[1][2] - 203) < EPSILON);
        assert(std::abs(result[1][3] - 218) < EPSILON);
        
        std::cout << "PASS\n";
    }

    static void testThreadBoundaries() {
        std::cout << "Testing thread boundary conditions... ";
        
        Matrix A(7, 7);
        Matrix B(7, 7);
        A.randomFill();
        B.randomFill();
        
        std::vector<int> threadCounts = {1, 2, 3, 4, 7, 8};
        
        Matrix reference = multiplySequential(A, B);
        
        for (int threads : threadCounts) {
            Matrix result = multiplyParallelSimple(A, B, threads);
            
            // Проверяем корректность результатов
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < 7; ++j) {
                    assert(std::abs(result[i][j] - reference[i][j]) < EPSILON);
                }
            }
        }
        
        std::cout << "PASS\n";
    }
};

const double MatrixTest::EPSILON = 1e-10;

int main() {
    try {
        MatrixTest::runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}