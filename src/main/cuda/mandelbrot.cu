#include <cuda_runtime_api.h>
#include "math.h"
#include "helpers.hpp"
#include "float.h"

typedef unsigned int uint;
using Pointf = Point<float>;
using Pointi = Point<uint>;

__device__ const float PI_F = 3.14159265358979f;
__device__ const uint MAX_SS_LEVEL = 256;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Mandelbrot: Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)
  
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Mandelbrot: Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

#define DEBUG_MODE
#ifdef DEBUG_MODE 
  #define ASSERT(x) assert(x)
#else 
  #define ASSERT(x) do {} while(0)
#endif

#ifndef CUDART_VERSION
  #error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 9000) //for cuda 9 and later, use __any_sync(__activemask(), predicate) instead, see Programming guide, B.13 for more details
  #define __ALL(predicate) __all_sync(__activemask(), predicate)
  #define __ANY(predicate) __any_sync(__activemask(), predicate)
#else
  #define __ALL(predicate) __all(predicate)
  #define __ANY(predicate) __any(predicate)
#endif



//Mandelbrot content, using standard mathematical terminology for Mandelbrot set definition, i.e.
//  f_n = f_{n-1}^2 + c
//  f_0 = 0
//  thus iteratively applying: f(z) = z*z * c
//  where z, c are complex numbers, with components denoted as
//    x ... for real part (corresponding to geometric x-axis)
//    y ... for imag part (corresponding to geometric y-axis)

template <class Real> __device__ __forceinline__
uint escape(uint dwell, Point<Real> c){
  Point<Real> z(0,0);
  Real zx_new;
  uint i = 0;
  while(i < dwell && z.x*z.x+z.y*z.y < 4){
      zx_new = z.x*z.x-z.y*z.y + c.x;
      z.y = 2*z.x*z.y + c.y; 
      z.x = zx_new;
      ++i;
  }
  return i;
}


/// Dispersion in this context is "Index of dispersion", aka variance-to-mean ratio. See https://en.wikipedia.org/wiki/Index_of_dispersion for more details
template <class Real> __device__ __forceinline__
Real computeDispersion(uint* data, uint dataLength, Real mean){
  uint n = dataLength;
  Real variance = 0;
  for(uint i = 0; i < dataLength; i++){
    //using numerically stable Two-Pass algorithm, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    variance += (data[i]-mean)*(data[i]-mean);
  }
  variance /= (n-1); 
  return variance / mean;
}


__device__ __forceinline__
bool isWithinRadius(uint idx_x, uint idx_y, uint width, uint height, uint radius, uint focus_x, uint focus_y){
  if(__sad(idx_x, focus_x, 0) > radius / 2) return false;
  if(__sad(idx_y, focus_y, 0) > radius / 2) return false;
  else return true;

  // if(idx_x < (width - radius)/2 || idx_y < (height-radius)/2) return false;
  // if(idx_x > (width + radius)/2 || idx_y > (height+radius)/2) return false;
  // else return true;
}

__device__ long seed;
/// Intended for debugging only
__device__ __forceinline__ uint simpleRandom(uint val){
    long long a = 1103515245;
    long long c = 12345;
    long long m = 4294967295l; //2**32 - 1
    seed = (a * (val+seed) + c) % m;
    return seed;
}

__device__ const uint WARP_SIZE_X = 8; //represents desired size of the recatangular warp (with respect to threadIdx). WARP_SIZE.x * WARP_SIZE.y should always be warpSize (32 for CUDA 9 and lower)
__device__ const uint WARP_SIZE_Y = 4;
  /// Computes indexes to a per-pixel acces of a 2D image, based on threadIdx and blockIdx.
  /// Morover, threads in a warp will be arranged in a rectangle (rather than in single line as with the naive implementation).
  /// The caller should always check if the returned value exceeded image width and height.
__device__ const Point<uint> getImageIndexes(){
  const uint threadID = threadIdx.x + threadIdx.y * blockDim.x;
  const uint warpH = WARP_SIZE_Y; // 2,4,8 are only reasonable values of warpH for the following formula
  const uint blockWidth = blockDim.x * warpH;
  ASSERT (blockDim.x == 32); //following formula works only when blockDim.x is 32 
  const uint inblock_idx_x = (-(threadID % (warpH * warpH)) + threadID % blockWidth) / warpH + threadID % warpH;
  const uint inblock_idx_y = (threadID / blockWidth) * warpH + (threadID / warpH) % warpH;
  const uint idx_x = blockDim.x * blockIdx.x + inblock_idx_x;
  const uint idx_y = blockDim.y * blockIdx.y + inblock_idx_y;
  // { //debug
  //   uint warpid = threadID / warpH;
  //   if(idx_x < 20 && idx_y < 20){
  //     //printf("bw:%u\n", blockWidth);
  //     printf("%u\t%u\t%u\t%u\t%u\n", threadIdx.x, threadIdx.y, threadID ,inblock_idx_x, inblock_idx_y);
  //   }
  // }
  return Point<uint>(idx_x, idx_y);
}

__device__ __forceinline__
pixel_info_t* getPtrToPixel(uint** array2D, long pitch, uint x, uint y){
  return (((pixel_info_t*)((char*)array2D + y * pitch)) + x);
}

/// param sampleCount: Maximum number of samples to take. Actual number of samples taken will be stored here before returning. If adaptiveSS==false, the value will not change.
template <class Real> __device__
uint sampleTheFractal(Pointi pixel, uint width, uint height, Rectangle<Real> image, uint dwell,uint & sampleCount, bool adaptiveSS, bool visualiseSS){
  const uint adaptiveTreshold = 10;
  uint r[adaptiveTreshold];

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  Point<Real> pixelSize = image.size() / Point<Real>(width, height);
  
  uint escapeTimeSum = 0;
  ASSERT (sampleCount <= MAX_SS_LEVEL);
  for(uint i = 0; i < sampleCount; i++){
    Point<Real> delta = Point<Real>(i / (Real) sampleCount);
    
    // c = {LBx, RTy} {+,-} ((pixel+delta) * pixelSize)
    const Point<Real> c = Point<Real>(image.left_bottom.x, image.right_top.y) +
      Point<Real>(1,-1) * (pixel.cast<Real>() + delta) * pixelSize;

    uint escapeTime = escape(dwell, c);
    escapeTimeSum += escapeTime;
    if(i < adaptiveTreshold){
      r[i] = escapeTime;
    }

    //todo this process should ideally adaptive, at least dynamic. Not predefined like.
    if(( i== 2 || i == adaptiveTreshold) && adaptiveSS){ //decide whether to continue with supersampling or not
      Real mean = escapeTimeSum / (i+1);
      Real dispersion = computeDispersion(r, i, mean);
      float dispersionMax = (i==2) ? 0.01 : 0.1;
      if(__ALL(dispersion <= dispersionMax )){
        sampleCount = i+1; //effectively disabling high SS and storing info about actual number of samples taken
      }
      //else we are on an chaotic edge, thus as many samples as possible are needed
    }
  }
  uint result = escapeTimeSum / sampleCount; 
  return result;
   

  //debug:
  // if(idx_x < 10 && idx_y < 10){
  //   printf("%f\t", randomSample);
  //   __syncthreads();
  //   if(idx_x == 0 && idx_y == 0)
  //     printf("\n");
  // }
} 

template <class Real> __device__ __forceinline__
void fractalRenderMain(uint** output, long outputPitch, uint width, uint height, Rectangle<Real> image, uint dwell, uint superSamplingLevel, bool adaptiveSS, bool visualiseSS, float* randomSamples, uint renderRadius, uint focus_x, uint focus_y, bool isDoublePrecision)
// todo: usporadat poradi paramateru, cudaXXObjects predavat pointrem, ne kopirovanim (tohle rozmyslet, mozna je to takhle dobre)
//  todo ma to fakt hodne pointeru, mnoho z nich je pritom pro vsechny launche stejny - nezdrzuje tohle? omezene registry a tak
{
  const Pointi idx = getImageIndexes();
  if(idx.x >= width || idx.y >= height) return;
  // if(idx.x == 0 && idx.y == 0){
  //   printf();
  // }
  //if(!isWithinRadius(idx.x, idx.y, width, height, renderRadius, focus_x, focus_y)) return;
  
  uint result = sampleTheFractal(idx, width, height, image, dwell, superSamplingLevel, adaptiveSS, visualiseSS);

  pixel_info_t* pOutput = getPtrToPixel(output, outputPitch, idx.x, idx.y);
  pOutput->value = result;
  pOutput->weight = (float) superSamplingLevel;
  ASSERT(pOutput->weight > 0);
}

//section exported global kernels:

extern "C" __global__
void fractalRenderMainFloat(uint** output, long outputPitch, uint width, uint height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, uint dwell, uint superSamplingLevel, bool adaptiveSS, bool visualiseSS, float* randomSamples, uint renderRadius, uint focus_x, uint focus_y){
  fractalRenderMain<float>(output, outputPitch, width, height, Rectangle<float>(left_bottom_x, left_bottom_y, right_top_x, right_top_y), dwell, superSamplingLevel, adaptiveSS, visualiseSS, randomSamples,  renderRadius, focus_x, focus_y, false);
}

extern "C" __global__
void fractalRenderMainDouble(uint** output, long outputPitch, uint width, uint height, double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y, uint dwell, uint superSamplingLevel, bool adaptiveSS, bool visualiseSS, float* randomSamples, uint renderRadius, uint focus_x, uint focus_y){
  fractalRenderMain<double>(output, outputPitch, width, height, Rectangle<double>(left_bottom_x, left_bottom_y, right_top_x, right_top_y), dwell, superSamplingLevel, adaptiveSS, visualiseSS, randomSamples,  renderRadius, focus_x, focus_y, true);

}

extern "C" __global__
void compose(uint** inputMain, long inputMainPitch, uint** inputBcg, long inputBcgPitch, cudaSurfaceObject_t surfaceOutput, uint width, uint height, cudaSurfaceObject_t colorPalette, uint paletteLength, uint dwell, uint mainRenderRadius, uint focus_x, uint focus_y){
  const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x >= width || idx_y >= height) return;

  /*
  const uint blurSize = 4;
  
  const uint convolution[blurSize][blurSize] = {
      //  {1,2,1},
      //  {2,4,2},
      //  {1,2,1}
      {0,0,0},
      {0,1,0},
      {0,0,0}
  };
  const uint convolutionDivisor = 1;

  uint sum = 0;
  #pragma unroll
  for(uint i = -blurSize/2; i < blurSize/2; i++){ 
    #pragma unroll
    for(uint j = -blurSize/2; j < blurSize/2; j++){
      uint x = max(0,min(width,idx_x + i));
      uint y = max(0,min(height,idx_y + j));
      uint* pInput1 = (uint*)((char*)input1 + y * input1pitch) + x;
      sum += (*pInput1) * convolution[i+blurSize/2][j+blurSize/2];
    }
  }
  uint result;
  result = sum / convolutionDivisor;
  */
  //choose result from one or two

  pixel_info_t* pResult;
  if(isWithinRadius(idx_x, idx_y, width, height, mainRenderRadius, focus_x, focus_y)){
    pResult = getPtrToPixel(inputMain, inputMainPitch, idx_x, idx_y);
  }else{
    pResult = getPtrToPixel(inputBcg, inputBcgPitch, idx_x, idx_y);
  }
  uint result = pResult->value;

  uint paletteIdx = paletteLength - (result % paletteLength) - 1;
//  ASSERT(paletteIdx >=0);
  ASSERT(paletteIdx < paletteLength);
  uint resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * sizeof(uint), 0);
  // if(result == dwell || result == dwell-1)
  //   resultColor = ColorsARGB::YELLOW;

  surf2Dwrite(resultColor, surfaceOutput, idx_x * sizeof(uint), idx_y);
}

extern "C" __global__
void blur(){}

extern "C" __global__
void fractalRenderUnderSampled(uint** output, long outputPitch, uint width, uint height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, uint dwell, uint underSamplingLevel)
{
  //work only at every Nth pixel:
  const uint idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * underSamplingLevel;
  const uint idx_y = (blockDim.y * blockIdx.y + threadIdx.y) * underSamplingLevel;
  if(idx_x >= width-underSamplingLevel || idx_y >= height-underSamplingLevel) return;
  
  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float pixelWidth = (right_top_x - left_bottom_x) / (float) width;
  float pixelHeight = (right_top_y - left_bottom_y) / (float) height;
  
  float cx = left_bottom_x + (idx_x)  * pixelWidth;
  float cy = right_top_y - (idx_y) * pixelHeight;

  uint escapeTime = escape(dwell, Pointf(cx, cy));

  for(uint x = 0; x < underSamplingLevel; x++){
    for(uint y = 0; y < underSamplingLevel; y++){
      //surf2Dwrite(resultColor, surfaceOutput, (idx_x + x) * sizeof(unsigned uint), (idx_y+y));
      pixel_info_t* pOutput = getPtrToPixel(output, outputPitch, idx_x+x, idx_y+y);
      pOutput->value = escapeTime;
      pOutput->weight = 1 / (float) underSamplingLevel;
    }
  }

}

struct big{
  uint a;
  uint b;
  uint c;
  uint d;
  uint e;
  uint f;
};

extern "C" __global__
void debug(big a, uint c){
  const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x == 0 && idx_y == 0){
    // printf("aa:\t%u\n",a.a);
    // printf("ab:\t%u\n",a.b);
    // printf("ac:\t%u\n",a.c);
    // printf("ad:\t%u\n",a.d);
    // printf("ae:\t%u\n",a.e);
    // printf("af:\t%u\n",a.f);
    // printf("c:\t%u\n",c);
  }
}

extern "C" __global__
void init(){

}

/// for given point <code>p</code> in the current image and given warping information, find cooridnates of the same point (=representing the same point in the fractal's complex plane) in the image being warped
/// @param p: the point whose warping origin is being returned
/// @param imageSize: width and height (in pixels) of current image
/// @param currentImage: rectangle representing the part of the complex plane that is being rendered
/// @param oldImage: rectangle representing the part of the complex plane that is being reused

template <class Real> __device__ __forceinline__
Point<Real> getWarpingOrigin(Point<Real> p, Point<Real> imageSize, Rectangle<Real> currentImage, Rectangle<Real> oldImage){

      Point<Real> size_current = currentImage.size();
      Point<Real> size_reused = oldImage.size();
      Point<Real> coeff = size_current / size_reused;

      Point<Real> deltaReal;    
      deltaReal.x = currentImage.left_bottom.x - oldImage.left_bottom.x;
      deltaReal.y = oldImage.right_top.y - currentImage.right_top.y;
      Point<Real> delta = deltaReal / size_current * imageSize;

      Point<Real> result = (p * coeff) + delta;
      return result;
}

__device__ float screenDistance = 60; //in cm; better be set by the user
__device__ __forceinline__
/// Returns how many samples this pixel should take, based on foveation.
/// Value is between 0 and maxSuperSamplingLevel.
/// Value in the focus will always be maxSuperSamplingLevel, values in the non-peripheral view will always be non-zero.
/// Returned value is the same for all pixels within a warp (the highest is taken).
uint getAdvisedSampleCount(Pointi pixel, Pointi focus, uint maxSuperSamplingLevel){
  //per-warp normalisation, i.e. set all pixels from a warp to same value
  pixel = pixel - (pixel % Pointi(WARP_SIZE_X, WARP_SIZE_Y));

  float pixelRealWidthInCm = 0.02652; //todo this value should probably be entered by the user. From http://www.prismo.ch/comparisons/desktop.php 
  float focusDistance = focus.cast<float>().distanceTo(pixel.cast<float>()) * pixelRealWidthInCm; //distance to focus, translated to cm
  /// visual angle for one eye, i.e possible values are from 0 to ~ 110
  float visualAngle = atan (focusDistance / screenDistance) * 180 / PI_F; //from https://en.wikipedia.org/wiki/Visual_angle
  
  //used model for (visualAngle -> relativeQuality): in (0,fovealViewLimit): a constant function that yields 1, in (fovealViewLimit, peripheralViewLimit): descenidng linear function from 1 to 0
  ASSERT(visualAngle >= 0); //arctan is positive for positive input and distance is always non-negative
  const float fovealViewLimit = 5.5; //in degrees, value from https://en.wikipedia.org/wiki/Peripheral_vision
  ///todo, this number is based on a paper. Based on my experience, it could be even smaller
  const float peripheralViewTreshold = 60;  //in degrees, value from https://en.wikipedia.org/wiki/Peripheral_vision
  float relativeQuality = (1/(fovealViewLimit-peripheralViewTreshold))*visualAngle+(-peripheralViewTreshold/(fovealViewLimit-peripheralViewTreshold)); 
  if(visualAngle <= fovealViewLimit) relativeQuality = 1;

  uint result = maxSuperSamplingLevel * relativeQuality;
  if(visualAngle <= peripheralViewTreshold)
    result = max(1, result); //always return at least 1 for pixels within the field of view
  return result;
}

extern "C" __global__
void fractalRenderReuseSamples(uint** output, long outputPitch, uint width, uint height, float a, float b, float c, float d, uint dwell, uint superSamplingLevel, bool adaptiveSS, bool visualiseSS, float* randomSamples, uint renderRadius, uint focus_x, uint focus_y, float p, float q, float r, float s, uint** input, long inputPitch, bool useFoveation, bool useSampleReuse){

  const Pointi idx = getImageIndexes();
  if(idx.x >= width || idx.y >= height) return;
  if(idx.x == 0 && idx.y == 0){
    printf("fractalRenderReuseSamples:\n");
  }
  ASSERT(idx.x < width);
  ASSERT(idx.y < height);
  Rectangle<float> image = Rectangle<float>(a,b,c,d);

  useSampleReuse = true;
  useFoveation = true;

  //sample reusal:
  bool reusingSample = false;
  uint reusalResult;
  float reusalWeight;
  if(useSampleReuse){
    const Pointf originf = getWarpingOrigin(Pointf(idx.x, idx.y),Pointf(width,height),image, Rectangle<float>(p,q,r,s));
    const Point<int> origin = Point<int>((int)round(originf.x), (int)round(originf.y)); //it is important to convert to signed int, not uint (because the value may be negative)
    if(origin.x < 0 || origin.x >= width || origin.y < 0 || origin.y >= height){
      reusalResult = 404;   //not-found error :)
      reusingSample = false;
    }
    else{
      pixel_info_t* pInput = getPtrToPixel(input, inputPitch, origin.x, origin.y);
      reusalResult = pInput->value;
      reusalWeight = (float) pInput->weight;
      reusingSample = true;
    }
  }
  
  //sample generation:
  uint sampleCount = superSamplingLevel;
  if (useFoveation)
    sampleCount = getAdvisedSampleCount(idx, Pointi(focus_x, focus_y), superSamplingLevel);
  if((!reusingSample || reusalWeight == 0) && sampleCount == 0)  sampleCount = 1; //at least one sample has to be taken somewhere
      //indeed, it may happen that reusalWeight == 0, because reusalWeight decreases in time (see reusedSampleDegradateCoeff)
  uint renderResult = sampleTheFractal(idx, width, height, image, dwell, sampleCount, adaptiveSS, visualiseSS);
  ASSERT(reusingSample || sampleCount > 0);

  //combine reused and generated samples:
  uint result;
  float resultWeight;
  if(reusingSample){
    if(sampleCount == 0){
      result = reusalResult;
      resultWeight = reusalWeight;    
    }else{
      const float reusedSampleDegradateCoeff = 0.65; // must be <=1
      reusalWeight *= reusedSampleDegradateCoeff;
      resultWeight = reusalWeight + sampleCount;    
      result = (reusalResult * reusalWeight + renderResult * sampleCount) / (resultWeight);
    }
  }else{
    result = renderResult;
    resultWeight = sampleCount;
  }

// //debug pravy dolni roh:
//       uint qqq = 16;
//   if(idx.x >= width-qqq || idx.y >= height-qqq)      
//       result = dwell;

  if(visualiseSS){
    if(reusingSample) ++sampleCount;
    result = sampleCount / float (superSamplingLevel) * 255;
  }

  pixel_info_t* pOutput = getPtrToPixel(output, outputPitch, idx.x, idx.y);
  pOutput->value = result;
  pOutput->weight = resultWeight;
  ASSERT(resultWeight > 0);
}


__device__ void printParams_debug(cudaSurfaceObject_t surfaceOutput, long outputDataPitch_debug, uint width, uint height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, uint dwell, uint** outputData_debug, cudaSurfaceObject_t colorPalette, uint paletteLength, float* randomSamples, uint superSamplingLevel, bool adaptiveSS, bool visualiseSS){
  const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x != 0 || idx_y != 0)
    return;
  printf("\n");
  printf("width:\t%u\n",width);
  printf("height:\t%u\n",height);
  printf("dwell:\t%u\n",dwell);
  printf("SS lvl:\t%u\n",superSamplingLevel);
}