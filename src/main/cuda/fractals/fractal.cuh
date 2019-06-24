#include "helpers.cuh"

/// The core fractal function. Find fractal value for given point,
/// but use at most maxIterations of finding cycle.
template <class Real> __device__
float computeFractal(unsigned int maxIterations, Point<Real> z);

/// Should find adequate color in the colorPalette and return it as unsigned int in RGBA (little endian, Red is the least significant).
/// Use surf2Dread() to retrieve data from the palette and see documentation 
/// or mandelbrot.cu for default implementaion.
/// The third argument is the value previsouly returned by iterate().
__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float iterationResult);
//The default implementation that you can reuse is following:

//       unsigned int iterationResult_i = round(iterationResult);
//       unsigned int paletteIdx = paletteLength - (iterationResult_i % paletteLength) - 1;
//       ASSERT(paletteIdx < paletteLength);
//       unsigned int resultColor;
//       surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
//       return resultColor;

__device__ void debugFractal();