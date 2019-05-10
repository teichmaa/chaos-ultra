#include "helpers.cuh"

/// The core fractal function. Find fractal value for given point,
/// but use at most maxIterations of finding cycle.
template <class Real> __device__
unsigned int iterate(unsigned int maxIterations, Point<Real> z);

/// Should find adequate color in the colorPalette and return it as unsigned int.
/// Use surf2Dread() to retrieve data from the palette and see documentation 
/// or mandelbrot.cu for default implementaion.
/// The third argument is the value previsouly returned by iterate().
__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, unsigned int iterationResult);

__device__ void debugFractal();