#ifndef __MandelbrotSet_H__
#define __MandelbrotSet_H__

#include "defines.h"

class MandelbrotSet {
public:
    //MandelbrotSet()=default;
    MandelbrotSet(int width,int height);
    ~MandelbrotSet();
    uint8_t calpixel(std::complex<double> c);
    void compute();
    uint8_t* get_data(){
        return data.data();
    }


private:
    std::vector<uint8_t> data;
    
    int width,height;
};

#endif