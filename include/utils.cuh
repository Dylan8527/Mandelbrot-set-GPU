#ifndef __UTILS_H__
#define __UTILS_H__

#include "defines.h"

struct WindowGuard final
{
  WindowGuard(GLFWwindow *&, const int width, const int height,
              const std::string &title);
  ~WindowGuard();
};

#define uint std::uint32_t
constexpr const uint BLOCKSIZE = 512;

// CUDA_CHECK_ERROR: check the return value of the CUDA runtime API call and exit the application if the call has failed.
#define CUDA_CHECK_ERROR(val) DO_CUDA_CHECK_ERROR((val), #val, __FILE__, __LINE__)


template <typename T>
static void DO_CUDA_CHECK_ERROR(T result, const char *const func, const char *const file, const int line) { 
    if (result) { 
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<uint>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE); 
    } 
}

// blockDim._ : number of threads in a thread block
// gridDim._ : number of thread blocks in a grid
// threadIdx._ : thread index in a thread block
// blockIdx._ : block index in a grid
#define cuda_foreach_uint(_, start, end) \
    for (uint _ = (start) + blockDim._ * blockIdx._ + threadIdx._; _ < (end); _ += blockDim._ * gridDim._) 

#endif