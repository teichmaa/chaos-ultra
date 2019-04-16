#include "helpers.hpp"

//Mandelbrot content, using standard mathematical terminology for Mandelbrot set definition, i.e.
//  f_n = f_{n-1}^2 + c
//  f_0 = 0
//  thus iteratively applying: f(z) = z*z * c
//  where z, c are complex numbers, with components denoted as
//    x ... for real part (corresponding to geometric x-axis)
//    y ... for imag part (corresponding to geometric y-axis)

template <class Real> __device__ __forceinline__
unsigned int escape(unsigned int maxIterations, Point<Real> c){
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