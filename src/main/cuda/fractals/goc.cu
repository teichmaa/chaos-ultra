#include "fractal.cuh"
#include <thrust/complex.h> 

using namespace thrust;

__constant__ int amplifier; 

template <class Real> __device__
complex<Real> gocihoMetoda(complex<Real> c, complex<Real> z){
  //42*Z^(-2)+C^7
  return 42 * 1 / (z*z) +(c*c*c*c*c*c*c);
}

template <class Real> __device__
float computeFractal(unsigned int maxIterations, Point<Real> x){

  if(x.x == 0 || x.y == 0)
    return 0;
  complex<Real> z = complex<Real>(x.x, x.y);
  complex<Real> c = z;
  unsigned int i = 0;
  while(i < maxIterations){
   z = gocihoMetoda(c,z);
   i++;
  }
  return (Point<Real>(z.real(), z.imag())).distanceTo(Point<Real>(0));

}

__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float iterationResult){
  unsigned int iterationResult_i = round(iterationResult);
  unsigned int paletteIdx = paletteLength - (iterationResult_i % paletteLength) - 1;
  ASSERT(paletteIdx < paletteLength);
  unsigned int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
  if(iterationResult < 35)
      resultColor &= 0xff00ffff;
  return resultColor;
}

__device__ void debugFractal(){
  printf("hello from goci\n");
}