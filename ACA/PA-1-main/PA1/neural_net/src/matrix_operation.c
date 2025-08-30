#include "matrix_operation.h"
#include <immintrin.h>

const int TILE_SIZE = 16;

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n,m);

	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j < m ; j++) {
			for(int q = 0; q < k; q++) {
				C(i, j) += A(i, q) * B(q, j);
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
        for (int z = 0; z < k; z++) {
            double a_val = A(i, z);

            for (int j = 0; j < m; j++) {
                C(i, j) += a_val * B(z, j);
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
            double sum = 0;

            for (int z = 0; z + UNROLL <= k; z += UNROLL) {
                sum += A(i, z) * B(z, j)
            		+ A(i, z + 1) * B(z + 1, j)
                	+ A(i, z + 2) * B(z + 2, j)
                	+ A(i, z + 3) * B(z + 3, j);
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
//----------------------------------------------------- Write your code here ----------------------------------------------------------------

     for (int i = 0; i < n; i += TILE_SIZE) {
        for (int j = 0; j < m; j += TILE_SIZE) {
            for (int z = 0; z < k; z += TILE_SIZE) {
                for (int ii = i; ii < i + TILE_SIZE; ii++) {
					for (int jj = j; jj < j + TILE_SIZE; jj++) {
						    for (int kk = z; kk < z + TILE_SIZE; kk++) {
								C(ii, jj) += A(ii, jj) * B(kk, jj);
						}
					}

					// Loop re ordering
                    // for (int kk = z; kk < z + TILE_SIZE; kk++) {
                    //     double a_val = A(ii, kk);
					//
                    //     for (int jj = j; jj < j + TILE_SIZE; jj++) {
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
    
    int width = 4; // 256 bit registers = 4 doubles

    for (int i = 0; i < n; i++) {
        for (int j = 0; j + width <= m; j += width) {
            __m256d c_chunk = _mm256_setzero_pd();

            for (size_t z = 0; z < k; z++) {
                __m256d a_chunk = _mm256_broadcast_sd(&A(i, z));
                __m256d b_chunk = _mm256_loadu_pd(&B(z, j));
                c_chunk = _mm256_add_pd(c_chunk, _mm256_mul_pd(a_chunk, b_chunk));
            }

            _mm256_storeu_pd(&C(i, j), c_chunk);
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
	for (int i = 0; i < rows; i += TILE_SIZE) {
        for (int j = 0; j < cols; j += TILE_SIZE) {
            for (int ii = i; ii < i + TILE_SIZE; ii++) {
                for (int jj = j; jj < j + TILE_SIZE; jj++) {
                    result(jj, ii) = A(ii, jj);
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}
