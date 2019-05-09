#include "fractal.cuh"
#include <thrust/complex.h> //documentation: https://thrust.github.io/doc/group__complex__numbers.html
#include <math.h>

using namespace thrust;
template <class Real> __device__
complex<Real> evalF(complex<Real> x){
    return (x * x * x) - 1;
}

template <class Real> __device__
complex<Real> evalFDerivative(complex<Real> x){
    return (3 * (x * x));
}

template <class Real> __device__
unsigned int iterate(unsigned int maxIterations, Point<Real> z){

    const complex<Real> root_a(1,0);
    const complex<Real> root_b(-0.5,0.86602540378); // 0.86602540378 = sqrt(3) / 2
    const complex<Real> root_c(-0.5,-0.86602540378);
    const Real tolerance = 0.000001;

    complex<Real> x(z.x,z.y);

    unsigned int i = 0;
    while(i < maxIterations){
        x = x - evalF(x) / evalFDerivative(x);
        ++i;
    }

    complex<Real> difference;
    difference = x - root_a;
    if(abs(difference.real()) < tolerance && abs(difference.imag()) < tolerance){
        return 1;
    }
    difference = x - root_b;
    if(abs(difference.real()) < tolerance && abs(difference.imag()) < tolerance){
        return 2;
    }
    difference = x - root_c;
    if(abs(difference.real()) < tolerance && abs(difference.imag()) < tolerance){
        return 3;
    }
    return 0;
}


__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, unsigned int iterationResult){
    switch(iterationResult){
        case 1:
            return ColorsRGBA::RED;
        case 2:
            return ColorsRGBA::GREEN;
        case 3:
            return ColorsRGBA::BLUE;
        default:
            return ColorsRGBA::BLACK;
    }
}


__device__ void debugFractal(){
    /* empty */
}
