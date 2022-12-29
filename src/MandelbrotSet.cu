#include "MandelbrotSet.cuh"

__global__ void computeKernel(uint8_t *data, int width, int height, double x_start, double x_finish, double y_start, double y_finish, vec3 *colortable_device);

MandelbrotSet::MandelbrotSet(int w, int h) : width(w), height(h)
{
    data_host.resize(width * height * 3);
    data_device = data_host;
    colortable_host = colormap();
    colortable_device = colortable_host;
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

    /*    float dx = (x_finish - x_start) / (width - 1);
        float dy = (y_finish - y_start) / (height - 1);

    #pragma omp parallel for
        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < height; ++y)
            {
                int offset = (width * y + x) * 3;

                int iteration = calpixel(std::complex<float>(x * dx + x_start, y * dy + y_start));
                vec3 color = colortable[iteration];
                data[offset + 0] = uint8_t(color.x * 255);
                data[offset + 1] = uint8_t(color.y * 255);
                data[offset + 2] = uint8_t(color.z * 255);
            }
        }*/
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
    if ((col < width) && (row < height))
    {
        double dx = (x_finish - x_start) / (width - 1);
        double dy = (y_finish - y_start) / (height - 1);
        int offset = (width * row + col) * 3;
        int count = 0;
        cuDoubleComplex c{col * dx + x_start, row * dy + y_start};
        cuDoubleComplex z=c;
        int max = 256;
        do
        {
            z = cuCadd(cuCmul(z, z), c);
            ++count;
        }while ((cuCabs(z) <= 2.0) && (count < max));

        int color_index = count == max ? max : count + 1 - log(log2(cuCabs(z)));
        vec3 color = colortable_device[color_index];
        data[offset + 0] = uint8_t(color.x*255);
        data[offset + 1] = uint8_t(color.y*255);
        data[offset + 2] = uint8_t(color.z*255);
        /*data[offset + 0] = color_index;
        data[offset + 1] = color_index;
        data[offset + 2] = color_index;*/
    }
    // return;
}
