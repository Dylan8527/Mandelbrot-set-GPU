#ifndef __CUDA_ARRAY_CUH__
#define __CUDA_ARRAY_CUH__
#include "utils.cuh"
#include "cuda_runtime.h"

/**
 * @brief
 *
 * @tparam T : type of data in the array
 */
template <typename T>
struct HostArray;

template <typename T>
struct DeviceArray;

template <typename T, T value>
__global__ void init_array_kernel(T *data, uint S);

template <typename T>
struct HostArray
{
    T *data;
    uint S;

    HostArray(uint S) : S(S) { data = (T *)std::malloc(sizeof(T) * S); }

    void free()
    {
        if (data)
        {
            std::free(data);
            data = nullptr;
        }
    }

    /**
     * @brief copy data from host to host
     *
     * @param host : host array
     */
    HostArray &operator=(const HostArray<T> &rhs)
    {
        if (&rhs == this)
            return *this; // self-assignment check expected
        // check the size of the host array
        if (S != rhs.S)
        {
            std::cerr << "HostArray::operator= : size mismatch" << std::endl;
            exit(1);
        }
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyHostToHost));
        return *this;
    }

    /**
     * @brief copy data from device to host
     *
     * @param device : device array
     */
    HostArray &operator=(const DeviceArray<T> &rhs)
    {
        if (S != rhs.S)
        {
            std::cerr << "HostArray::operator= : size mismatch" << std::endl;
            exit(1);
        }
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyDeviceToHost));
        return *this;
    }

    T &operator[](uint i) { return data[i]; }
    const T &operator[](uint i) const { return data[i]; }

    [[nodiscard]] constexpr uint size() const { return S; }
};

template <typename T>
__global__ void init_array_kernel(T *data, uint S, T value)
{
    cuda_foreach_uint(x, 0, S) { data[x] = value; }
}

template <typename T>
struct DeviceArray
{
    T *data;
    uint S;

    DeviceArray(uint S) : S(S)
    {
        CUDA_CHECK_ERROR(cudaMalloc(&data, sizeof(T) * S));
    }

    void free()
    {
        if (data)
        {
            CUDA_CHECK_ERROR(cudaFree(data));
            data = nullptr;
        }
    }

    /**
     * @brief copy data from host to device
     *
     * @param host : host array
     */
    DeviceArray &operator=(const HostArray<T> &rhs)
    {
        if (S != rhs.S)
        {
            std::cerr << "DeviceArray::operator= : size mismatch" << std::endl;
            exit(1);
        }
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyHostToDevice));
        return *this;
    }

    /**
     * @brief copy data from device to device
     *
     * @param device : device array
     */
    DeviceArray &operator=(const DeviceArray<T> &rhs)
    {
        if (&rhs == this)
            return *this; // self-assignment check expected
        if (S != rhs.S)
        {
            std::cerr << "DeviceArray::operator= : size mismatch" << std::endl;
            exit(1);
        }
        CUDA_CHECK_ERROR(
            cudaMemcpy(data, rhs.data, sizeof(T) * S, cudaMemcpyDeviceToDevice));
        return *this;
    }

    __device__ __forceinline__ T &operator[](uint i) { return data[i]; }
    __device__ __forceinline__ const T &operator[](uint i) const { return data[i]; }

    [[nodiscard]] constexpr __host__ __device__ __forceinline__ uint
    size() const
    {
        return S;
    }

    /**
     * @brief initialize the array with a value
     *
     * @param value : value to initialize the array with
     */
    void init(T value)
    {
        init_array_kernel<T><<<ceil(S / static_cast<double>(BLOCKSIZE)), BLOCKSIZE>>>(data, S, value);
    }
};

#endif