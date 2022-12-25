#include <MandelbrotSet.h>

MandelbrotSet::MandelbrotSet(int w, int h) : width(w), height(h)
{
    data.resize( width * height * 3);
    #pragma omp parallel for
    for (int i = 0; i < data.size(); i++)
        data[i] = static_cast<uint8_t>(rand());

}

MandelbrotSet::~MandelbrotSet()
{
}

uint8_t MandelbrotSet::calpixel(std::complex<double> c)
{
    int count = 0;
    std::complex<double> z(0, 0);
    double tmp, lengthsq;
    int max = 256;
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

void MandelbrotSet::compute()
{
    double x_start = -2.0;
	double x_fin = 1.0;
	double y_start = -1.0;
	double y_fin = 1.0;
    double dx = (x_fin - x_start)/(width - 1);
	double dy = (y_fin - y_start)/(height - 1);
#pragma omp parallel for
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            int offset = (width * y + x) * 3;
            data[offset + 0] = calpixel(std::complex<double>(x * dx+x_start, y * dy+y_start));
            data[offset + 1]=data[offset + 0];
            data[offset + 2]=data[offset + 0];
        }
    }
}