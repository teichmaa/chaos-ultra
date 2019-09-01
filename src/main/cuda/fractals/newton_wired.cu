#include "fractal.cuh"
#include <thrust/complex.h> //documentation: https://thrust.github.io/doc/group__complex__numbers.html
#include <math.h>

using namespace thrust;

template <class Real> __device__
complex<Real> newtonMethod(complex<Real> x){
    complex<Real> x_pow_2 = x * x;
    complex<Real> x_pow_3 = x_pow_2 * x;
    
    complex<Real> f_eval_x =  x_pow_3 - 1;
    complex<Real> f_derivative_eval_x = 3 * x_pow_2;

    return x - (f_eval_x / f_derivative_eval_x);
}

template <class Real> __device__
unsigned int convergenceRoot(complex<Real> x){
    
    const complex<Real> root_a(1,0);
    const complex<Real> root_b(-0.5,0.86602540378); // 0.86602540378 = sqrt(3) / 2
    const complex<Real> root_c(-0.5,-0.86602540378);

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
float computeFractal(unsigned int maxIterations, Point<Real> z){

    complex<Real> x(z.x,z.y);

    unsigned int i = 0;
    unsigned int convergenceCheckTreshold = 10; //most of the plane segments generally converges after 10 iterations
    while(i < maxIterations){
        x = newtonMethod(x);
        ++i;

        if(i == convergenceCheckTreshold){
            unsigned int root = convergenceRoot(x);
            if(root != 0)   //if already converged
                return root;
            //else
            convergenceCheckTreshold += maxIterations / 10; //go up by 10 percent
        }
    }
    return convergenceRoot(x);
}


__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float iterationResult){
    unsigned int iterationResult_i = round(iterationResult);
    switch(iterationResult_i){
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
