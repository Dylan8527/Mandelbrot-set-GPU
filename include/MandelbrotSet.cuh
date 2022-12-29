#ifndef __MandelbrotSet_H__
#define __MandelbrotSet_H__

#include "defines.h"
#include "cuda_array.cuh"
#include "cuComplex.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


class MandelbrotSet {
public:
    //MandelbrotSet()=default;
    MandelbrotSet(int width,int height);
    ~MandelbrotSet();
    int calpixel(std::complex<double> c);
    void compute(double x_start,double x_finish,double y_start,double y_finish);
    uint8_t* get_data(){
        return data_host.data();
    }
    std::vector<vec3> colormap(vec3 theta=vec3(.1f, .15f, .25f),int color_size=1<<8);

    __global__ void calpixel_kernel(float dx, float dy, float x_start, float y_start, int *x, int *y, );
private:
    thrust::host_vector<uint8_t> data_host;
    thrust::device_vector<uint8_t> data_device;
    thrust::host_vector<vec3> colortable_host;
    thrust::device_vector<vec3> colortable_device;
    int width,height;
    const int max_iterations=256;
    
};

#endif