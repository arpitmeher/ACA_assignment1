#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
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

    vector<vector<float>> outputs;

    for (size_t i = 0; i < offsets.size(); i++) {
        int start = offsets[i];
        int end = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();
        vector<float> sum(embedding_dim, 0);

        for (int j = start; j < end; j++) {
            int next = j + prefetch_distance;

            if (next < end) {
                auto next_item_addr = (const char*) &embedding_table[input[next] * embedding_dim];
                _mm_prefetch(next_item_addr, _MM_HINT_T0);
            }

            auto current_data = &embedding_table[input[j] * embedding_dim];

            for (int j = 0; j < embedding_dim; j++) {
                sum[j] += current_data[j];
            }
        }

        outputs.push_back(sum);
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();

    //----------------------------------------------------- Write your code here ----------------------------------------------------------------

    // 256 bit register
    
    // vector<vector<float>> outputs;
    // int chunk_width = embedding_dim / 8; // 128/8 = 16 (256 bit register can hold 8 floats at once)

    // for (size_t i = 0; i < offsets.size(); i++) {
    //     int start = offsets[i];
    //     int end = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();
    //     vector<float> sum(embedding_dim, 0);
    //     vector<__m256> simd_registers(chunk_width);

    //     for (int i = 0; i < chunk_width; i++) {
    //         simd_registers[i] = _mm256_setzero_ps();
    //     }

    //     for (int i = start; i < end; i++) {
    //         auto data = &embedding_table[input[i] * embedding_dim];
            
    //         for (int j = 0; j < chunk_width; j++) {
    //             simd_registers[j] = _mm256_add_ps(simd_registers[j], _mm256_loadu_ps(data + j * 8));
    //         }
    //     }

    //     for (int i = 0; i < chunk_width; i++) {
    //         _mm256_storeu_ps(&sum[i * 8], simd_registers[i]);
    //     }

    //     outputs.push_back(sum);
    // }
    
    // 128 bit register
    
    vector<vector<float>> outputs;
    int chunk_width = embedding_dim / 4; // 128/4 = 16 (128 bit register can hold 4 floats at once)

    for (size_t i = 0; i < offsets.size(); i++) {
        int start = offsets[i];
        int end = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();
        vector<float> sum(embedding_dim, 0);
        vector<__m128d> simd_registers(chunk_width);

        for (int i = 0; i < chunk_width; i++) {
            simd_registers[i] = _mm_setzero_ps();
        }

        for (int i = start; i < end; i++) {
            auto data = &embedding_table[input[i] * embedding_dim];
            
            for (int j = 0; j < chunk_width; j++) {
                simd_registers[j] = _mm_add_ps(simd_registers[j], _mm_loadu_ps(data + j * 8));
            }
        }

        for (int i = 0; i < chunk_width; i++) {
            _mm_storeu_ps(&sum[i * 8], simd_registers[i]);
        }

        outputs.push_back(sum);
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();

    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> outputs;
    int chunk_width = embedding_dim / 8;

    for (size_t i = 0; i < offsets.size(); i++) {
        int start = offsets[i];
        int end = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> sum(embedding_dim, 0.0f);
        vector<__m256> simd_registers(chunk_width);
        
        for (int i = 0; i < chunk_width; i++) {
            simd_registers[i] = _mm256_setzero_ps();
        }

        for (int i = start; i < end; i++) {
            int next = i + prefetch_distance;
            
            if (next < end) {
                auto next_item_addr = &embedding_table[input[next] * embedding_dim];
                _mm_prefetch(next_item_addr, _MM_HINT_T0);
            }

            auto data_ptr = &embedding_table[input[i] * embedding_dim];
            
            for (int j = 0; j < chunk_width; j++) {
                simd_registers[j] = _mm256_add_ps(simd_registers[j], _mm256_loadu_ps(data_ptr + j * 8));
            }
        }

        for (int i = 0; i < chunk_width; i++) {
            _mm256_storeu_ps(&sum[i * 8], simd_registers[i]);
        }

        outputs.push_back(sum);
    }

    //-------------------------------------------------------------------------------------------------------------------------------------------

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching and SIMD: " << duration.count() << " microseconds.";

    return duration.count();
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
    long long time_with_simd = run_with_simd(embedding_table, input, offsets);
    // Run emb with software prefetching and simd
    long long time_with_prefetch_simd = run_with_prefetching_simd(embedding_table, input, offsets);


    // Compute speedup
    double speedup1 = static_cast<double>(time_without_prefetch) / time_with_prefetch;
    double speedup2 = static_cast<double>(time_without_prefetch) / time_with_simd;
    double speedup3 = static_cast<double>(time_without_prefetch) / time_with_prefetch_simd;
    cout << fixed << setprecision(3);
    cout << "\n\nSpeedup (with software prefetching) = " << speedup1 << "x\n";
    cout << "Speedup (with simd) = " << speedup2 << "x\n";
    cout << "Speedup (with software prefetching and simd) = " << speedup3 << "x\n";

    return 0;
}
