#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h> 
#include <emmintrin.h>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int embedding_table_size = 1000000;
const int embedding_dim = 128;
const int input_size = 720;
const int num_bags = 20;

const int prefetch_distance = 8;

int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

long long run_with_prefetching(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
     vector<vector<float>> output;

    // for (size_t b = 0; b < offsets.size(); ++b) {
    //     int start = offsets[b];
    //     int end = (b + 1 < offsets.size()) ? offsets[b + 1] : input.size();

    //     vector<float> bag_embedding(embedding_dim, 0.0f);

    //     for (int t = start; t < end; ++t) {
    //         int ahead = t + prefetch_distance;
    //         if (ahead < end) {
    //             const float* ptr = &embedding_table[input[ahead] * embedding_dim];
    //             _mm_prefetch((const char*)ptr, _MM_HINT_T0);
    //         }

    //         const float* data_ptr = &embedding_table[input[t] * embedding_dim];
            
    //         for (int d = 0; d < embedding_dim; ++d) {
    //             bag_embedding[d] += data_ptr[d];
    //         }
    //     }

    //     output.push_back(std::move(bag_embedding));
    // }
    
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    // vector<vector<float>> output;
    // const int vec_iters = embedding_dim / 8; // 128/8 = 16

    // for (size_t b = 0; b < offsets.size(); ++b) {
    //     int start = offsets[b];
    //     int end = (b + 1 < offsets.size()) ? offsets[b + 1] : input.size();
    //     vector<float> bag_embedding(embedding_dim, 0.0f);
    //     vector<__m256> acc(vec_iters);

    //     for (int vi = 0; vi < vec_iters; ++vi) acc[vi] = _mm256_setzero_ps();

    //     for (int t = start; t < end; ++t) {
    //         const float* data_ptr = &embedding_table[input[t] * embedding_dim];
    //         for (int vi = 0; vi < vec_iters; ++vi) {
    //             __m256 v = _mm256_loadu_ps(data_ptr + vi * 8);
    //             acc[vi] = _mm256_add_ps(acc[vi], v);
    //         }
    //     }

    //     for (int vi = 0; vi < vec_iters; ++vi) {
         
    //     //    __m128d _mm_add_pd (__m128d a, __m128d b);
    //         _mm256_storeu_ps(&bag_embedding[vi * 8], acc[vi]);
    //     }

    //     output.push_back(std::move(bag_embedding));
    // }
    // //-------------------------------------------------------------------------------------------------------------------------------------------
    
    // auto end = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";

    // return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    //    vector<vector<float>> output;
    // const int vec_iters = embedding_dim / 8;

    // for (size_t b = 0; b < offsets.size(); ++b) {
    //     int start = offsets[b];
    //     int end = (b + 1 < offsets.size()) ? offsets[b + 1] : input.size();

    //     vector<float> bag_embedding(embedding_dim, 0.0f);
    //     vector<__m256> acc(vec_iters);
    //     for (int vi = 0; vi < vec_iters; ++vi) acc[vi] = _mm256_setzero_ps();

    //     for (int t = start; t < end; ++t) {
    //         // Prefetch rows ahead
    //         int ahead = t + prefetch_distance;
    //         if (ahead < end) {
    //             const float* prefetch_ptr = &embedding_table[input[ahead] * embedding_dim];
    //             _mm_prefetch((const char*)prefetch_ptr, _MM_HINT_T0);
    //         }

    //         const float* data_ptr = &embedding_table[input[t] * embedding_dim];
    //         for (int vi = 0; vi < vec_iters; ++vi) {
    //             __m256 v = _mm256_loadu_ps(data_ptr + vi * 8);
    //             acc[vi] = _mm256_add_ps(acc[vi], v);
    //         }
    //     }

    //     for (int vi = 0; vi < vec_iters; ++vi) {
    //         _mm256_storeu_ps(&bag_embedding[vi * 8], acc[vi]);
    //     }

    //     output.push_back(std::move(bag_embedding));
    // }
    
    // //-------------------------------------------------------------------------------------------------------------------------------------------
    
    // auto end = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // cout << "\nTime WITH software prefetching and SIMD: " << duration.count() << " microseconds.";

    // return duration.count();
}


long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    
    return duration.count();
}

int main() {
    // Prepare embedding table
    vector<float> embedding_table(embedding_table_size * embedding_dim);
    for (auto& val : embedding_table) {
        val = static_cast<float>(random_int(embedding_table_size));
    }

    // Input indices
    vector<int> input(input_size);
    for (auto& idx : input) {
        idx = random_int(embedding_table_size);
    }

    // Offsets
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) {
        offsets.push_back((input_size * i) / num_bags);
    }

    // Run naive code 
    long long time_without_prefetch = naive_emb(embedding_table, input, offsets);
    
    // ---------- Flush Cache Before Running Prefetching ----------
    for (size_t i = 0; i < embedding_table.size(); i += 16) {
        _mm_clflush(&embedding_table[i]);
    }
    _mm_mfence();
    
    // Run emb with software prefetching 
    long long time_with_prefetch = run_with_prefetching(embedding_table, input, offsets);
    // Run emb with simd 
    // long long time_with_simd = run_with_simd(embedding_table, input, offsets);
    // Run emb with software prefetching and simd
    // long long time_with_prefetch_simd = run_with_prefetching_simd(embedding_table, input, offsets);
    

    // Compute speedup
    double speedup1 = static_cast<double>(time_without_prefetch) / time_with_prefetch;
    double speedup2 = static_cast<double>(time_without_prefetch) / 1;
    double speedup3 = static_cast<double>(time_without_prefetch) / 1;
    cout << fixed << setprecision(3);
    cout << "\n\nSpeedup (with software prefetching) = " << speedup1 << "x\n";
    cout << "Speedup (with simd) = " << speedup2 << "x\n";
    cout << "Speedup (with software prefetching and simd) = " << speedup3 << "x\n";

    return 0;
}

