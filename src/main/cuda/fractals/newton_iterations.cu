#include "fractal.cuh"
#include <thrust/complex.h> //documentation: https://thrust.github.io/doc/group__complex__numbers.html
#include <math.h>

using namespace thrust;

__constant__ Point<double> roots[3]; 
__constant__ double coefficients[4]; 
__constant__ int colorMagnifier; 


template <class Real> __device__
complex<Real> newtonMethod(complex<Real> x){
    complex<Real> x_pow_2 = x * x;
    complex<Real> x_pow_3 = x_pow_2 * x;
    
    complex<Real> f_eval_x =  coefficients[0] + 
                              coefficients[1] * x + 
                              coefficients[2] * x_pow_2 + 
                              coefficients[3] * x_pow_3 ;
    complex<Real> f_derivative_eval_x = coefficients[1] + 
                                        coefficients[2] * 2 * x + 
                                        coefficients[3] * 3 * x_pow_2 ;

    return x - (f_eval_x / f_derivative_eval_x);
}

template <class Real> __device__
unsigned int convergenceRoot(complex<Real> x){
    
    const complex<Real> root_a(roots[0].x,roots[0].y);
    const complex<Real> root_b(roots[1].x,roots[1].y); 
    const complex<Real> root_c(roots[2].x,roots[2].y);

    const Real tolerance = 0.0001;
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

template <class Real> __device__
unsigned int iterate(unsigned int maxIterations, Point<Real> z){

    complex<Real> x(z.x,z.y);

    unsigned int i = 0;
    while(i < maxIterations){
        x = newtonMethod(x);
        ++i;
        unsigned int root = convergenceRoot(x);
        if(root != 0)
            return i;
    }
    return i;
}


__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, unsigned int iterationResult){
    unsigned int paletteIdx = paletteLength - (iterationResult * colorMagnifier % paletteLength) - 1;
    ASSERT(paletteIdx < paletteLength);
    unsigned int resultColor;
    surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
    return resultColor;
}



__device__ void debugFractal(){
    /* empty */
}





