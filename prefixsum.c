// Fill a buffer with random float values and measure run time of prefix sum implementations

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <tmmintrin.h>
#include <math.h>
#include <time.h>


// prefix sum with conversion to uint8_t using SSE3 intrinsics
//
// Taken from Raph Levien's font-rs (https://github.com/google/font-rs)
// https://github.com/google/font-rs/blob/master/src/accumulate.c
//
// n (size of the buffers) needs to be a multiple of 4.
void prefix_sum_to_u8_sse(const float *in, uint8_t *out, uint32_t n) {
    __m128 offset = _mm_setzero_ps();
    __m128i mask = _mm_set1_epi32(0x0c080400);
    __m128 sign_mask = _mm_set1_ps(-0.f);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_load_ps(&in[i]);
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x = _mm_add_ps(x, _mm_shuffle_ps(_mm_setzero_ps(), x, 0x40));
        x = _mm_add_ps(x, offset);
        __m128 y = _mm_andnot_ps(sign_mask, x);  // fabs(x)
        y = _mm_min_ps(y, _mm_set1_ps(1.0f));
        y = _mm_mul_ps(y, _mm_set1_ps(255.0f));
        __m128i z = _mm_cvtps_epi32(y);
        z = _mm_shuffle_epi8(z, mask);
        _mm_store_ss((float *)&out[i], (__m128)z);
        offset = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

// prefix sum with conversion to uint8_t using plain C
void prefix_sum_to_u8_naive(const float *in, uint8_t *out, uint32_t n) {
    float sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += in[i];
        float x = fabsf(sum);
        x = x < 1.0f ? x * 255.0f : 255.0f;

        // round to nearest (like the SSE version)
        out[i] = (uint8_t)(x + 0.5f);
    }
}


typedef void (*prefix_sum_func_t)(const float *in, uint8_t *out, uint32_t n);

// run func several times and print average run time
void benchmark(const float *input_buffer,
               uint32_t size,
               prefix_sum_func_t func,
               const char *name,
               uint8_t *output_buffer)
{
    printf("\n%s\n", name);
    fflush(stdout);

    clock_t ticks = 0;
    double seconds = 0.0;
    int i = 0;
    for (i = 0; i < 128; i++) {
        clock_t start = clock();
        func(input_buffer, output_buffer, size);
        clock_t end = clock();
        ticks += end - start;
        seconds = ticks / (double)CLOCKS_PER_SEC;
        if (i >= 4 && seconds > 1.0) {
            break;
        }
    }

    printf("ran %d times\n", i);
    printf("avg time: %f secs\n", seconds / (double)(i + 1));

    // print some values from the output buffer
    for (int k = 0; k < 5 && k < size; k++) {
        printf("buffer[%d] = %d\n", k, output_buffer[k]);
    }
    printf("buffer[%d] = %d\n", size - 1, output_buffer[size - 1]);

    fflush(stdout);
}

int main(int argc, char **argv) {
    // buffer size
    uint32_t size = 1024 * 1024 * 128;

    // parse first command line argument as size
    if (argc == 2) {
        char *end = NULL;
        long new_size = strtol(argv[1], &end, 10);
        if (argv[1] != end && new_size >= 4 && new_size <= (1024 * 1024 * 1024)) {
            // size should be a multiple of 4
            size = new_size - (new_size % 4);
        } else {
            printf("Could not accept first argument: Not a valid buffer size.\n");
            return 1;
        }
    }

    printf("buffer size: %d\n", size);
    fflush(stdout);

    float *buffer = (float*)malloc(size * sizeof(float));

    printf("\nfilling buffer...");
    fflush(stdout);

    // fill buffer with random values
    srand(time(0));
    for (int i = 0; i < size; i++) {
        buffer[i] = (rand() % 2000 - 1000) * 0.001;
    }

    printf(" Done.\n");
    fflush(stdout);


    // measure time for prefix_sum_sse
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_u8_sse, "SSE", output_buffer);
        free(output_buffer);
    }

    // measure time for prefix_sum_naive
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_u8_naive, "NAIVE", output_buffer);
        free(output_buffer);
    }


    free(buffer);

    return 0;
}
