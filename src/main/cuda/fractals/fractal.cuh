#include "helpers.cuh"

/// The core fractal function. Find fractal value for given point,
/// but use at most maxIterations of finding cycle.
/// @param maxIterations: an upped bound of the complexity of the computation, usually interpreted as maximal number of iteration of a complex map
/// @param z: a point, representing the complex number to compute the fractal/chaotic value at.
template <class Real> __device__
float computeFractal(unsigned int maxIterations, Point<Real> z);

/// Find adequate color in the colorPalette and return it as unsigned int in RGBA (little endian, Red is the least significant).
/// <br>
/// This method is intended to give the user broader control of fractal coloring.
/// Use surf2Dread() to retrieve data from the palette and see documentation or mandelbrot.cu for default implementaion.
/// @param colorPalette: cuda read-only surface, typed as 2D texture, representing a 2D texture with exactly one row; being logically one-dimensional.
/// @param computationResult: Usually the value previsouly returned by computeFractal(). Note that it need NOT to be exact value from the last rendering but may be for example average of the last and previous renderings.
__device__ __forceinline__
unsigned int colorize(cudaSurfaceObject_t colorPalette, unsigned int paletteLength, float computationResult);
//The default implementation that you can reuse is following:

//       //index the palette with (computationResult), modulo paletteLength:
//       unsigned int computationResult_i = round(computationResult);
//       unsigned int paletteIdx = paletteLength - (computationResult_i % paletteLength) - 1;
//       ASSERT(paletteIdx < paletteLength);
//       unsigned int resultColor;
//       surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(unsigned int), 0);
//       return resultColor;

__device__ void debugFractal();