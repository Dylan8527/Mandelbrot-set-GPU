#ifndef __MandelbrotSet_H__
#define __MandelbrotSet_H__

#include "defines.h"
#include "cuda_array.cuh"
#include "cuComplex.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define colortable_size 1<<12

class MandelbrotSet {
public:
    //MandelbrotSet()=default;
    MandelbrotSet(int width,int height);
    ~MandelbrotSet();
    int calpixel(std::complex<double> c);
    void compute(double x_start,double x_finish,double y_start,double y_finish);
    void escapetime_based_algorithm(double x_start,double x_finish,double y_start,double y_finish);
    void basic_algorithm(double x_start,double x_finish,double y_start,double y_finish);
    uint8_t* get_data(){
        return data_host.data();
    }
    std::vector<vec3> colormap(vec3 theta=vec3(.85, .0, .15),int color_size=colortable_size);
    void update_colormap(vec3 theta);
    void update_parameter(int new_maxiter, double new_ncycle, double new_stripe_s, double new_stripe_sig, double new_step_s);

private:
    thrust::host_vector<uint8_t> data_host;
    thrust::device_vector<uint8_t> data_device;
    thrust::host_vector<vec3> colortable_host;
    thrust::device_vector<vec3> colortable_device;
    int width,height;
    const int max_iterations=256;

    int maxiter=500;         /* maximal number of iteration */
    double ncycle=32;           /* number of iteration before cycling the colortable */
    double stripe_s=0;       /* frequency parameter of stripe average coloring.(set 0 for no stripes) */
    double stripe_sig=.9;    /* memory parameter of stripe average coloring */
    double step_s=0;         /* frequency parameter of step coloring.(set 0 for no steps) */
    double light[7] = {45., 45., .75, .2, .5, .5, 20};/* light vector: angle azimuth [0-360], angle elevation [0-90],
                  opacity [0,1], k_ambiant, k_diffuse, k_spectral, shininess*/
};

#endif