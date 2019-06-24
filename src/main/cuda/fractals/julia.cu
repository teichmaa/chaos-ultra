#include "fractal.cuh"

__constant__ double julia_c[2]; 

template <class Real> __device__ __forceinline__
float computeFractal(unsigned int maxIterations, Point<Real> z){
  Point<Real> c((Real) julia_c[0],(Real) julia_c[1]);
  Real zx_new;
  unsigned int i = 0;
  while(i < maxIterations && z.x*z.x+z.y*z.y < 4){
      zx_new = z.x*z.x-z.y*z.y + c.x;
      z.y = 2*z.x*z.y + c.y; 
      z.x = zx_new;
      ++i;
  }
  return i;
}

__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float iterationResult){
  unsigned int iterationResult_i = round(iterationResult);
  unsigned int paletteIdx = paletteLength - (iterationResult_i % paletteLength) - 1;
  ASSERT(paletteIdx < paletteLength);
  unsigned int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
  return resultColor;
}

__device__ void debugFractal(){
  printf("hello from julia\n");
}