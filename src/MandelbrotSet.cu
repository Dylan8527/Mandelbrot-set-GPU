#include "MandelbrotSet.cuh"


MandelbrotSet::MandelbrotSet(int w, int h) : width(w), height(h)
{
    data.resize(width * height * 3);
    colortable=colormap();
}

MandelbrotSet::~MandelbrotSet()
{
}

std::vector<vec3> MandelbrotSet::colormap(vec3 theta, int color_size)
{
    float start = 0, finish = 1;
    float dx = (finish - start) / color_size;
    std::vector<vec3>  colors(color_size+1);
    for (int i = 0; i <= color_size; ++i)
    {
        vec3 color;
        color = (vec3(dx * i) + theta) * 2.0f * PI;
        color.x = 0.5f + std::sin(color.x);
        color.y = 0.5f + std::sin(color.y);
        color.z = 0.5f + std::sin(color.z);
        colors[i]=color;
    }
    return colors;
}

int MandelbrotSet::calpixel(std::complex<float> c)
{
    int count = 0;
    std::complex<float> z(0, 0);
    float tmp, lengthsq;
    int max = max_iterations;
    do
    {
        tmp = z.real() * z.real() - z.imag() * z.imag() + c.real();
        z.imag(2.0 * z.real() * z.imag() + c.imag());
        z.real(tmp);
        lengthsq = std::norm(z);
        ++count;
    } while ((lengthsq < 4.0) && (count < max));
    return count;
}



void MandelbrotSet::compute(float x_start,float x_finish,float y_start,float y_finish)
{

    float dx = (x_finish - x_start) / (width - 1);
    float dy = (y_finish - y_start) / (height - 1);

#pragma omp parallel for
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            int offset = (width * y + x) * 3;
            /*data[offset + 0] = calpixel(std::complex<float>(x * dx + x_start, y * dy + y_start));
            data[offset + 1] = data[offset + 0];
            data[offset + 2] = data[offset + 0];*/

            int iteration=calpixel(std::complex<float>(x * dx + x_start, y * dy + y_start));
            vec3 color=colortable[iteration];
            data[offset+0]=uint8_t(color.x*255);
            data[offset+1]=uint8_t(color.y*255);
            data[offset+2]=uint8_t(color.z*255);
        }
    }
}