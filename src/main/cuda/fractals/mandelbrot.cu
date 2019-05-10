#include "fractal.cuh"

//Mandelbrot content, using standard mathematical terminology for Mandelbrot set definition, i.e.
//  f_n = f_{n-1}^2 + c
//  f_0 = 0
//  thus iteratively applying: f(z) = z*z * c
//  where z, c are complex numbers, with components denoted as
//    x ... for real part (corresponding to geometric x-axis)
//    y ... for imag part (corresponding to geometric y-axis)

template <class Real> __device__ __forceinline__
unsigned int iterate(unsigned int maxIterations, Point<Real> c){
  Point<Real> z(0,0);
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
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, unsigned int iterationResult){
  unsigned int paletteIdx = paletteLength - (iterationResult % paletteLength) - 1;
  ASSERT(paletteIdx < paletteLength);
  unsigned int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
  return resultColor;
}

__device__ void debugFractal(){
  printf("hello from mandelbrot\n");
}