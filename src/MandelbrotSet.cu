#include "MandelbrotSet.cuh"

__global__ void computeKernel(uint8_t *data, int width, int height, double x_start, double x_finish, double y_start, double y_finish, vec3 *colortable_device);
/* 1. basic algorithm */
__global__ void basic_kernel(uint8_t *data, 
                                uint width,
                                uint height, 
                                double x_start, 
                                double x_finish, 
                                double y_start, 
                                double y_finish,
                                int maxiter);

/* 2. escape time based algorithm */
__global__ void escapetime_kernel(uint8_t *data, 
                                uint width,
                                uint height, 
                                double x_start, 
                                double x_finish, 
                                double y_start, 
                                double y_finish,
                                int maxiter,
                                vec3* colortable,
                                int ncycle,
                                double stripe_s,
                                double stripe_sig,
                                double step_s,
                                double phi,
                                double theta,
                                double opacity,
                                double k_ambient,
                                double k_diffuse,
                                double k_specular,
                                double shininess);

__device__ void smooth_iter(cuDoubleComplex c, 
                            int maxiter, 
                            double stripe_s,
                            double stripe_sig,
                            double &niter,
                            double &stripe_a,
                            double &dem,
                            cuDoubleComplex &normal);

__device__ __forceinline__ void color_pixel(uint8_t *data, 
                                            double niter,
                                            double stripe_a,
                                            double step_s,
                                            double dem,
                                            cuDoubleComplex normal,
                                            vec3* colortable,
                                            int ncycle,
                                            double phi,
                                            double theta,
                                            double opacity,
                                            double k_ambient,
                                            double k_diffuse,
                                            double k_specular,
                                            double shininess);

__device__ __forceinline__ void overlay(double x, 
                                        double y,
                                        double gamma,
                                        double &output); 

__device__ __forceinline__ void blinn_phong_lighting(cuDoubleComplex normal, 
                                                    double phi,
                                                    double theta,
                                                    double opacity,
                                                    double k_ambient,
                                                    double k_diffuse,
                                                    double k_specular,
                                                    double shininess,
                                                    double &brightness);

MandelbrotSet::MandelbrotSet(int w, int h) : width(w), height(h)
{
    data_host.resize(width * height * 3);
    data_device = data_host;
    colortable_host = colormap();
    colortable_device = colortable_host;

    ncycle=sqrt(ncycle);
    light[0]=2*PI*light[0]/360.;
    light[1]=PI/2.*light[1]/90.;

}

MandelbrotSet::~MandelbrotSet()
{
}

std::vector<vec3> MandelbrotSet::colormap(vec3 theta, int color_size)
{
    double start = 0, finish = 1;
    double dx = (finish - start) / color_size;
    std::vector<vec3> colors(color_size + 1);
    // #pragma omp parallel for
    for (int i = 0; i <= color_size; ++i)
    {
        vec3 color;
        color = (vec3(dx * i) + theta) * 2.0f * PI;
        color.x = 0.5f + std::sin(color.x);
        color.y = 0.5f + std::sin(color.y);
        color.z = 0.5f + std::sin(color.z);
        colors[i] = color;
    }
    return colors;
}

void MandelbrotSet::update_colormap(vec3 theta) {
    colortable_host = colormap(theta);
    colortable_device = colortable_host;
}

int MandelbrotSet::calpixel(std::complex<double> c)
{
    int count = 0;
    std::complex<double> z = c;
    double tmp, lengthsq;
    int max = max_iterations;
    do
    {
        /*tmp=(z.real()*z.real()-z.imag()*z.imag())+c.real();
        z.imag(2.0f*z.real()*z.imag()+c.imag());
        z.real(tmp);*/
        z = z * z + c;
        lengthsq = std::norm(z);
        ++count;
    } while ((lengthsq < 4.0) && (count < max));
    return count;
}

#define TILE_WIDTH 32
void MandelbrotSet::compute(double x_start, double x_finish, double y_start, double y_finish)
{
    dim3 dimGrid(ceil((double)width / TILE_WIDTH), ceil((double)height / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    uint8_t *dataptr = thrust::raw_pointer_cast(&data_device[0]);
    vec3 *colortableptr = thrust::raw_pointer_cast(&colortable_device[0]);
    computeKernel<<<dimGrid, dimBlock>>>(dataptr, width, height, x_start, x_finish, y_start, y_finish, colortableptr);
    data_host = data_device;
}

__global__ void computeKernel(uint8_t *data, int width, int height, double x_start, double x_finish, double y_start, double y_finish, vec3 *colortable_device)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    static const double esc_radius = 2;

    if ((col < width) && (row < height))
    {
        double dx = (x_finish - x_start) / (width - 1);
        double dy = (y_finish - y_start) / (height - 1);
        int offset = (width * row + col) * 3;
        int count = 0;
        cuDoubleComplex c{col * dx + x_start, row * dy + y_start};
        cuDoubleComplex z{0, 0};
        int max = 256;
        do
        {
            z = cuCadd(cuCmul(z, z), c);
            ++count;
        }while ((cuCabs(z) <= esc_radius) && (count < max));

        int color_index = count == max ? max : count + 1 - log(log2(cuCabs(z)));
        vec3 color = colortable_device[color_index];
        data[offset + 0] = uint8_t(color.x*255);
        data[offset + 1] = uint8_t(color.y*255);
        data[offset + 2] = uint8_t(color.z*255);
    }
}

void MandelbrotSet::basic_algorithm(double x_start, double x_finish, double y_start, double y_finish)
{
    uint S = width * height;
    uint8_t *dataptr = thrust::raw_pointer_cast(&data_device[0]);
    basic_kernel<<<ceil(S/512.), 512>>>(dataptr, width, height, x_start, x_finish, y_start, y_finish, maxiter);
    data_host = data_device;
}

void MandelbrotSet::escapetime_based_algorithm(double x_start, double x_finish, double y_start, double y_finish) {
    uint S = width * height;
    uint8_t *dataptr = thrust::raw_pointer_cast(&data_device[0]);
    vec3 *colortableptr = thrust::raw_pointer_cast(&colortable_device[0]);
    escapetime_kernel<<<ceil(S/512.), 512>>>(dataptr, width, height, x_start, x_finish, y_start, y_finish, maxiter, colortableptr, ncycle, stripe_s, stripe_sig, step_s, light[0], light[1], light[2], light[3], light[4], light[5], light[6]);
    cudaDeviceSynchronize();
    data_host = data_device;
}

__global__ void basic_kernel(uint8_t *data, 
                                uint width,
                                uint height, 
                                double x_start, 
                                double x_finish, 
                                double y_start, 
                                double y_finish,
                                int maxiter){
    uint S = width * height;
    cuda_foreach_uint(x, 0, S) {
        uint row = x / width;
        uint col = x % width;
        double dx = (x_finish - x_start) / (width - 1);
        double dy = (y_finish - y_start) / (height - 1);
        int offset = (width * row + col) * 3;
        cuDoubleComplex z{0, 0};
        cuDoubleComplex c{col * dx + x_start, row * dy + y_start};
        data[offset] = 0;
        data[offset + 1] = 0;
        data[offset + 2] = 0;
        for (int i = 0; i < maxiter; ++i) {
            z = cuCadd(cuCmul(z, z), c);
            if (cuCabs(z) > 2) {
                data[offset] = 255;
                data[offset + 1] = 255;
                data[offset + 2] = 255;
                break;
            }
        }
    }
}

__global__ void escapetime_kernel(uint8_t *data, 
                                uint width,
                                uint height, 
                                double x_start, 
                                double x_finish, 
                                double y_start, 
                                double y_finish,
                                int maxiter,
                                vec3* colortable,
                                int ncycle,
                                double stripe_s,
                                double stripe_sig,
                                double step_s,
                                double phi,
                                double theta,
                                double opacity,
                                double k_ambient,
                                double k_diffuse,
                                double k_specular,
                                double shininess){
    uint S = width * height;
    cuda_foreach_uint(x, 0, S) {
        uint row = x / width;
        uint col = x % width;
        double dx = (x_finish - x_start) / (width - 1);
        double dy = (y_finish - y_start) / (height - 1);
        int offset = (width * row + col) * 3;
        cuDoubleComplex z{0, 0};
        cuDoubleComplex c{col * dx + x_start, row * dy + y_start};

        double niter, stripe_a, dem;
        cuDoubleComplex normal;
        smooth_iter(c, maxiter, stripe_s, stripe_sig, niter, stripe_a, dem, normal);
        if(niter > 0) {
            color_pixel(data+offset, niter, stripe_a, step_s, dem, normal, colortable, ncycle, phi, theta, opacity, k_ambient, k_diffuse, k_specular, shininess);
        }
        else{
            data[offset] = 0;
            data[offset + 1] = 0;
            data[offset + 2] = 0;
        }
    }
}
                            
__device__ void smooth_iter(cuDoubleComplex c, 
                            int maxiter, 
                            double stripe_s,
                            double stripe_sig,
                            double &niter,
                            double &stripe_a,
                            double &dem,
                            cuDoubleComplex &normal) {
    cuDoubleComplex z{0, 0};
    cuDoubleComplex dz{1, 0};
    cuDoubleComplex two{2, 2};
    cuDoubleComplex one{1, 0};

    double esc_radius = 1e5; 

    bool is_stripe = (stripe_s > 0) && (stripe_sig > 0);
    double stripe_tt;
    double modz;

    int n = 0;
    for(n = 0; n < maxiter; ++n) {
        dz = cuCadd(cuCmul(two, cuCmul(z, dz)), one);
        z = cuCadd(cuCmul(z, z), c);
        if(is_stripe) {
            stripe_tt = (sin(stripe_s*atan2(cuCimag(z), cuCreal(z)))+1) / 2.;
        }
        modz = cuCabs(z);
        if (modz > esc_radius) {
            double log_ratio = log(modz) / log(esc_radius);
            double smooth_i =  1 - log(log_ratio) / log(2.);
            if(is_stripe) {
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig-1)) + stripe_tt * smooth_i * (1 - stripe_sig));
                stripe_a = stripe_a / (1 - pow(stripe_sig, n) * (1 + smooth_i * (stripe_sig-1)));
            }
            normal = cuCdiv(z, dz);
            dem = modz * log(modz) / cuCabs(dz) / 2;
            niter = n + smooth_i;
            break;
        }

        if (is_stripe) {
            stripe_a = stripe_a * stripe_sig + stripe_tt * (1 - stripe_sig);
        }
    }
    if(n == maxiter) {
        niter = 0;
        stripe_a = 0;
        dem = 0;
        normal = {0, 0};
    }
}

__device__ __forceinline__ void color_pixel(uint8_t *data, 
                                            double niter,
                                            double stripe_a,
                                            double step_s,
                                            double dem,
                                            cuDoubleComplex normal,
                                            vec3* colortable,
                                            int ncycle,
                                            double phi,
                                            double theta,
                                            double opacity,
                                            double k_ambient,
                                            double k_diffuse,
                                            double k_specular,
                                            double shininess) {
    int ncol = (colortable_size) - 1;
    double iter = (double)((int)sqrt(niter) % ncycle) / ncycle;
    int col_i = round(iter * ncol);

    double brightness;
    blinn_phong_lighting(normal, phi, theta, opacity, k_ambient, k_diffuse, k_specular, shininess, brightness);

    dem = -log(dem) / 12;
    dem = 1 / (1 + exp(-10 * ((2*dem-1)/2)));

    int nshader = 0;
    double shader = 0;

    if(stripe_a > 0) {
        nshader += 1;
        shader = shader + stripe_a;
    } 
    if(step_s > 0) {
        step_s = 1/step_s;
        col_i = round((iter - (int)iter % (int)step_s)*ncol);
        double x = (int)iter % (int)step_s / step_s;
        double light_step = 6*(1-pow(x,5)-pow(1-x,100))/10;
        step_s = step_s/8;
        x = (int)iter % (int)step_s / step_s;
        double light_step2 = 6*(1-pow(x,5)-(1-x,30))/10;
        double light_step_mixed;
        overlay(light_step2, light_step, 1, light_step_mixed);
        nshader += 1;
        shader = shader + light_step_mixed;
    }
    if(nshader > 0) {
        double light;
        shader = shader / nshader;
        overlay(brightness, shader, 1, light);
        brightness = light * (1-dem) + dem * brightness;
    }
    vec3& color = colortable[col_i];
    double r, g, b;
    overlay(color.x, brightness, 1, r);
    overlay(color.y, brightness, 1, g);
    overlay(color.z, brightness, 1, b);
    r = fmaxf(0, fminf(1, r));
    g = fmaxf(0, fminf(1, g));
    b = fmaxf(0, fminf(1, b));

    data[0] = uint8_t(r * 255);
    data[1] = uint8_t(g * 255);
    data[2] = uint8_t(b * 255);

}

__device__ __forceinline__ void overlay(double x, 
                                        double y,
                                        double gamma,
                                        double &output) {
    if(y < 0.5)
        output = 2 * x * y;
    else
        output = 1 - 2 * (1 - x) * (1 - y);
    output = output * gamma + x * (1 - gamma);

}

__device__ __forceinline__ void blinn_phong_lighting(cuDoubleComplex normal, 
                                                    double phi,
                                                    double theta,
                                                    double opacity,
                                                    double k_ambient,
                                                    double k_diffuse,
                                                    double k_specular,
                                                    double shininess,
                                                    double &brightness) {
    cuDoubleComplex modn{cuCabs(normal), cuCabs(normal)};                                                    
    normal = cuCdiv(normal, modn);
    // Diffuse
    double ldiff = cuCreal(normal)*cos(phi)*cos(theta) + cuCimag(normal)*sin(phi)*cos(theta) + sin(theta);
    ldiff = ldiff / (1 + sin(theta));

    // Specular
    double theta_half = (PI/2. + theta) / 2.;
    double lspec = cuCreal(normal)*cos(phi)*sin(theta_half) + cuCimag(normal)*sin(phi)*sin(theta_half) + cos(theta_half);
    lspec = lspec / (1 + cos(theta_half));
    lspec = pow(lspec, shininess);

    brightness = k_ambient + k_diffuse * ldiff + k_specular * lspec;
    brightness = brightness * opacity + (1 - opacity)/2.;

}