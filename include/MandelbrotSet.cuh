#ifndef __MandelbrotSet_H__
#define __MandelbrotSet_H__

#include "defines.h"
#include "cuda_array.cuh"

class MandelbrotSet {
public:
    //MandelbrotSet()=default;
    MandelbrotSet(int width,int height);
    ~MandelbrotSet();
    int calpixel(std::complex<float> c);
    void compute(float x_start,float x_finish,float y_start,float y_finish);
    uint8_t* get_data(){
        return data.data();
    }
    std::vector<vec3> colormap(vec3 theta=vec3(.0f, .15f, .25f),int color_size=1<<12);

    __global__ void calpixel_kernel(float dx, float dy, float x_start, float y_start, int *x, int *y, );
private:
    std::vector<uint8_t> data;
    std::vector<vec3> colortable;
    int width,height;
    const int max_iterations=256;
    
};

#endif