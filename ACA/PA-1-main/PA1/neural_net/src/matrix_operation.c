#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    for (int i = 0; i < n; i++) {
        for (int l = 0; l < k; l++) {
            double a_val = A(i, l);
            for (int j = 0; j < m; j++) {
                C(i, j) += a_val * B(l, j);
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 4;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
   
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            int l = 0;

            for (; l + UNROLL <= k; l += UNROLL) {
                sum += A(i, l)     * B(l, j)
                     + A(i, l + 1) * B(l + 1, j)
                     + A(i, l + 2) * B(l + 2, j)
                     + A(i, l + 3) * B(l + 3, j);
            }

            C(i, j) = sum;
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 32;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------

     for (int i = 0; i < n; i += T) {
        for (int j = 0; j < m; j += T) {
            for (int l = 0; l < k; l += T) {
                for (int ii = i; ii < i + T; ii++) {
					for (int jj = j; jj < j + T; jj++) {
						    for (int kk = l; kk < l + T; kk++) {
								C(ii, jj) += A(ii, jj) * B(kk, jj);
						}
					}
                    // for (int kk = l; kk < l + T; kk++) {
                    //     double a_val = A(ii, kk);
                    //     for (int jj = j; jj < j + T; jj++) {
                    //         C(ii, jj) += a_val * B(kk, jj);
                    //     }
                    // }
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    const int VEC = 4; // 256-bit AVX = 4 doubles

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j + VEC <= m; j += VEC) {
            __m256d c_vec = _mm256_setzero_pd();

            for (size_t l = 0; l < k; l++) {
                __m256d a_val = _mm256_broadcast_sd(&A(i, l));
                __m256d b_vec = _mm256_loadu_pd(&B(l, j));
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_val, b_vec));
            }

            _mm256_storeu_pd(&C(i, j), c_vec);
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	// for (size_t i = 0; i < rows; ++i) {
	// 	for (size_t j = 0; j < cols; ++j) {
	// 		result(j, i) = A(i, j);
	// 	}
	// }

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
const int T = 32;
	for (size_t i = 0; i < rows; i += T) {
        for (size_t j = 0; j < cols; j += T) {
            for (size_t ii = i; ii < i + T; ii++) {
                for (size_t jj = j; jj < j + T; jj++) {
                    result(jj, ii) = A(ii, jj);
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}
