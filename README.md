# Mandelbrot-set-GPU
CS121 Fall2022 Final Project——The implementation of Mandelbrot set on GPU



## python

​	We try Mandelbrot set algorithm first by python and check its correctness.



## docs

​	This directory includes our reference documents.

**Reference website:**

《分形理论与应用》中的Mandelbrot set https://zhuanlan.zhihu.com/p/392574615


### Compile

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
cd Release
./main.exe
```