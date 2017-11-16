// Fill a buffer with random float values and measure run time of prefix sum implementations

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tmmintrin.h>


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

// prefix sum with conversion to ARGB pixels using SSE2 intrinsics
//
// Most lines are taken from Raph Levien's font-rs.
// https://github.com/google/font-rs/blob/master/src/accumulate.c
//
// n (size of the buffers) needs to be a multiple of 4.
void prefix_sum_to_argb_sse_float(const float *in, uint8_t *out, uint32_t n) {
    uint32_t *out_32 = (uint32_t*)out;

    __m128 offset = _mm_setzero_ps();
    __m128 sign_mask = _mm_set1_ps(-0.f);

    __m128i alpha = _mm_set1_epi32(0xff000000);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_load_ps(&in[i]);
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x = _mm_add_ps(x, _mm_shuffle_ps(_mm_setzero_ps(), x, 0x40));
        x = _mm_add_ps(x, offset);
        __m128 y = _mm_andnot_ps(sign_mask, x);  // fabs(x)
        y = _mm_min_ps(y, _mm_set1_ps(1.0f));
        y = _mm_mul_ps(y, _mm_set1_ps(255.0f));

        // distribute value [0, 255] to RGB bytes with float intrinsics.

        // round
        y = _mm_cvtepi32_ps(_mm_cvtps_epi32(y));

        // 0x010101 == 65793
        y = _mm_mul_ps(y, _mm_set1_ps(65793.0f));

        // add alpha component
        __m128i z = _mm_add_epi32(_mm_cvtps_epi32(y), alpha);

        _mm_storeu_si128((__m128i*)&out_32[i], z);
        offset = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

// prefix sum with conversion to ARGB pixels using SSE2 intrinsics.
// Uses integer intrinsics for ARGB conversion.
//
// Most lines are taken from Raph Levien's font-rs.
// https://github.com/google/font-rs/blob/master/src/accumulate.c
//
// n (size of the buffers) needs to be a multiple of 4.
void prefix_sum_to_argb_sse_int(const float *in, uint8_t *out, uint32_t n) {
    uint32_t *out_32 = (uint32_t*)out;

    __m128 offset = _mm_setzero_ps();
    __m128 sign_mask = _mm_set1_ps(-0.f);

    __m128i alpha = _mm_set1_epi32(0xff000000);
    __m128i colors_r = _mm_set1_epi32(0x00000100);
    __m128i colors_gb = _mm_set1_epi32(0x00000101);
    for (int i = 0; i < n; i += 4) {
        __m128 x = _mm_load_ps(&in[i]);
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x = _mm_add_ps(x, _mm_shuffle_ps(_mm_setzero_ps(), x, 0x40));
        x = _mm_add_ps(x, offset);
        __m128 y = _mm_andnot_ps(sign_mask, x);  // fabs(x)
        y = _mm_min_ps(y, _mm_set1_ps(1.0f));
        y = _mm_mul_ps(y, _mm_set1_ps(255.0f));

        // distribute value [0, 255] to RGB bytes with integer intrinsics.

        __m128i a = _mm_cvtps_epi32(y);
        __m128i b = _mm_add_epi32(_mm_slli_epi32(_mm_mullo_epi16(a, colors_r), 8), alpha);
        __m128i z = _mm_add_epi32(_mm_mullo_epi16(a, colors_gb), b);

        _mm_storeu_si128((__m128i*)&out_32[i], z);
        offset = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 3, 3));
    }
}

// prefix sum with conversion to ARGB pixels using plain C
void prefix_sum_to_argb_naive(const float *in, uint8_t *out, uint32_t n) {
    // 1 pixel consists of 4 uint8_t values
    uint32_t *out_32 = (uint32_t*)out;

    uint32_t colors = (1 << 16) + (1 << 8) + 1;
    uint32_t alpha = 255 << 24;

    float sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        sum += in[i];
        float x = fabsf(sum);
        x = x < 1.0f ? x * 255.0f : 255.0f;

        // round to nearest (like the SSE version)
        out_32[i] = (uint32_t)(x + 0.5f) * colors + alpha;
    }
}


// Compare the two given buffers with length n.
bool compare_buffers(uint8_t *buf_a, uint8_t *buf_b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (buf_a[i] != buf_b[i]) {
            printf("Buffers not equal at element[%d]: %d != %d\n", i, buf_a[i], buf_b[i]);
            return false;
        }
    }
    return true;
}

// Compare the two given buffers: buf_u8 (with length n) and buf_argb (with
// length n*4).
bool compare_u8_to_argb_buffers(uint8_t *buf_u8, uint8_t *buf_argb, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint8_t val = buf_u8[i];
        if (val != buf_argb[i * 4] ||
            val != buf_argb[i * 4 + 1] ||
            val != buf_argb[i * 4 + 2] ||
            buf_argb[i * 4 + 3] != 255)
        {
            printf("Buffers not equal at element %d: [%d, %d, %d, 255] != [%d, %d, %d, %d]\n",
                    i, val, val, val,
                    buf_argb[i * 4],
                    buf_argb[i * 4 + 1],
                    buf_argb[i * 4 + 2],
                    buf_argb[i * 4 + 3]);
            return false;
        }
    }
    return true;
}


typedef void (*prefix_sum_func_t)(const float *in, uint8_t *out, uint32_t n);

// Runs func several times and measures run time. Prints the average, minimum
// value and standard deviation of those run times.
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
    double timings[128];
    double min_time = -1.0;
    int i = 0;
    for (i = 0; i < 128; i++) {
        clock_t start = clock();
        func(input_buffer, output_buffer, size);
        clock_t end = clock();

        ticks += end - start;
        seconds = ticks / (double)CLOCKS_PER_SEC;
        timings[i] = (end - start) / (double)CLOCKS_PER_SEC;

        if (i == 0) {
            min_time = timings[i];
        } else {
            min_time = fmin(min_time, timings[i]);
        }

        if (i >= 4 && seconds > 1.0) {
            break;
        }
    }

    printf("ran %d times\n", i);

    double avg_time = seconds / (double)(i + 1);
    printf("avg time: %f secs\n", avg_time);
    printf("min time: %f secs\n", min_time);

    // report standard deviation of timings
    {
        double diffs = 0.0;
        for (int k = 0; k <= i; k++) {
            diffs += pow(timings[k] - avg_time, 2.0);
        }

        double std_dev = sqrt(diffs / (double)i);
        printf("std deviation: %f\n", std_dev);
    }

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


    // measure time for prefix_sum_to_u8_naive
    uint8_t *comparison_buffer = (uint8_t*)malloc(size * sizeof(uint8_t));
    benchmark(buffer, size, prefix_sum_to_u8_naive, "NAIVE_U8", comparison_buffer);

    // measure time for prefix_sum_to_u8_sse
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_u8_sse, "SSE_U8", output_buffer);
        compare_buffers(comparison_buffer, output_buffer, size);
        free(output_buffer);
    }

    // measure time for prefix_sum_to_argb_naive
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * 4 * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_argb_naive, "NAIVE_ARGB", output_buffer);
        compare_u8_to_argb_buffers(comparison_buffer, output_buffer, size);
        free(output_buffer);
    }

    // measure time for prefix_sum_to_argb_sse_float
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * 4 * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_argb_sse_float, "SSE_ARGB_FLOAT", output_buffer);
        compare_u8_to_argb_buffers(comparison_buffer, output_buffer, size);
        free(output_buffer);
    }

    // measure time for prefix_sum_to_argb_sse_int
    {
        uint8_t *output_buffer = (uint8_t*)malloc(size * 4 * sizeof(uint8_t));
        benchmark(buffer, size, prefix_sum_to_argb_sse_int, "SSE_ARGB_INT", output_buffer);
        compare_u8_to_argb_buffers(comparison_buffer, output_buffer, size);
        free(output_buffer);
    }

    free(comparison_buffer);
    free(buffer);

    return 0;
}
