#include "fractal.cuh"
#include <thrust/complex.h> 

using namespace thrust;

__constant__ int amplifier; 

template <class Real> __device__
float iterate(unsigned int maxIterations, Point<Real> x){

  

  return (int) ((x).manhattanDistanceTo(Point<Real>(0,0)) * amplifier );
}

__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float iterationResult){
  unsigned int iterationResult_i = round(iterationResult);
  unsigned int paletteIdx = paletteLength - (iterationResult_i * 128 % paletteLength) - 1;
  ASSERT(paletteIdx < paletteLength);
  unsigned int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
  return resultColor;
}

__device__ void debugFractal(){
  printf("hello from test\n");
}