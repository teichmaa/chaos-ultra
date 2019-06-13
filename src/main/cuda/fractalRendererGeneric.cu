#include <cuda_runtime_api.h>
#include <math.h>
#include <float.h>
#include "helpers.cuh"
#include "fractals/fractal.cuh"

typedef unsigned int uint;
using Pointf = Point<float>;
using Pointi = Point<uint>;

constexpr float PI_F = 3.14159265358979f;
constexpr uint MAX_SUPER_SAMPLING = 64;

__constant__ bool VISUALIZE_SAMPLE_COUNT = false;

extern "C" __global__
void init(){

}

/// Get fractional part of a number.
template <class Real> __device__ __forceinline__
Real frac(Real x){
  return x - trunc(x);
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
  //TODO debug only: use original values instead
  // const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x; 
  // const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  return Point<uint>(idx_x, idx_y);
}

/**
 * Get a correct pointer to pixel_info of given pixel in given cuda 2D array
 */
__device__ __forceinline__
pixel_info_t* getPtrToPixelInfo(pixel_info_t** array2D,const long pitch,const Point<uint> pixel){
  return ((pixel_info_t*) ((char*)array2D + pixel.y * pitch)) + pixel.x;
}

/// sampleCount will be trimmed to be lesser or equal to sampleCount100Percent. 
/// samples will be colorized in linear grayscale from black (0) to white (sampleCount100Percent)
__device__ __forceinline__
uint colorizeSampleCount(uint sampleCount, uint sampleCount100Percent = 64){
  sampleCount = min(sampleCount, sampleCount100Percent);
  float sampleCountRel = sampleCount / (float) sampleCount100Percent;
  color_t result;
  result.rgba.r = 255 * sampleCountRel;
  result.rgba.g = 255 * sampleCountRel;
  result.rgba.b = 255 * sampleCountRel;
  result.rgba.a = 0xff;
  return result.intValue;
}

/// param pixel: pixel to sample
/// param gridSize: size of the grid of pixels (usually the output texture).
/// param image: segment of the complex plane that is to be rendered as an image.gridSize}.
/// param sampleCount: Maximum number of samples to take. Actual number of samples taken will be stored here before returning. The float value will always be round to the nearest int. If adaptiveSS==false, the value will not change (apart from rounding). 
template <class Real> __device__
uint sampleTheFractal(Pointi pixel, Pointi gridSize, Rectangle<Real> image, uint maxIterations,float & sampleCountF, bool adaptiveSS){
  if(sampleCountF < 1){
    sampleCountF = 0;
    return 0;
  }
  uint sampleCount = round(sampleCountF);

  const uint adaptiveTreshold = 10;
  uint samples[adaptiveTreshold];

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  Point<Real> pixelSize = image.size() / gridSize.cast<Real>();
  
  uint result;
  uint escapeTimeSum = 0;
 ASSERT (sampleCount <= MAX_SUPER_SAMPLING);
  for(uint i = 0; i < sampleCount; i++){

    //todo tady bych radeji neco jako deltas[sampleCount]; createDeltas(& deltas) nebo tak. Zkratka abych nesamploval takhle primitivne po diagonale, ale lip. Random sampling na to samozrejme je nejlepsi, ale to asi fakt ma overhead? Nevim, jestli mam cas to merit.
    Point<Real> delta = Point<Real>(i / (Real) sampleCount); //TODO TODO TODO PLS PLS PLS
    //todo jako druhou deltu zvolit pixelSize / 2.
    
    const Point<Real> flipYAxis = Point<Real>(1,-1);
    const Point<Real> image_left_top = Point<Real>(image.left_bottom.x, image.right_top.y);

    /// a point in the complex plane that is to be rendered
    // c = {LT} {+,-} ((pixel+delta) * pixelSize)
    const Point<Real> c = image_left_top + flipYAxis * (pixel.cast<Real>() + delta) * pixelSize;

    uint escapeTime = iterate(maxIterations, c);
    escapeTimeSum += escapeTime;
    if(i < adaptiveTreshold){
      samples[i] = escapeTime;
    }

    //todo refactor
    if(adaptiveSS && ( i > 0 && i < adaptiveTreshold) || (i == sampleCount / 2) ){ //decide whether to continue with supersampling or not

      Real mean = escapeTimeSum / (i+1);
      Real dispersion = computeDispersion(samples, i, mean);       //todo toto do else vetve

      constexpr float epsilon = 0.01;

      if(i == 1 && __ALL(samples[0] == samples[1])){
        sampleCount = i+1; //terminating this cycle and storing info about actual number of samples taken
        //result = 255 * 0;  // blue  //this is for hypothesis testing only
      } 
      else if(__ALL(dispersion < epsilon)){ // uniform distribution
        sampleCount = i+1; //terminating this cycle and storing info about actual number of samples taken
        //result = 255 * 6;  //dark blue  //this is for hypothesis testing only
      }
      else if(__ALL(dispersion <= 1) && i >= sampleCount / 2){ //binomial distribution - not that much chaotic -- take up to half of max allowed samples
        sampleCount = i+1; //terminating this cycle and storing info about actual number of samples taken
       // result = 255 * 2;  // green  //this is for hypothesis testing only
      }
      //else dispersion > 1 -- chaos: as many samples as possible are needed
      else{
        //result = 255 * 4; //red  //this is for hypothesis testing only
      }
    }
  }
  result = escapeTimeSum / sampleCount;
  sampleCountF = sampleCount; //write to the input-output param
  return result;
   

  //debug:
  // if(idx_x < 10 && idx_y < 10){
  //   printf("%f\t", randomSample);
  //   __syncthreads();
  //   if(idx_x == 0 && idx_y == 0)
  //     printf("\n");
  // }
} 

__device__ const uint USE_ADAPTIVE_SS_FLAG_MASK = (1 << 0);
__device__ const uint USE_FOVEATION_FLAG_MASK = (1 << 2);
__device__ const uint USE_SAMPLE_REUSE_FLAG_MASK = (1 << 3);
__device__ const uint IS_ZOOMING_FLAG_MASK = (1 << 4);
__device__ const uint ZOOMING_IN_FLAG_MASK = (1 << 5);

__device__ const uint visualityAmplifyCoeff = 10;

/// param image: segment of the complex plane that is to be rendered as an image
template <class Real> __device__ __forceinline__
void fractalRenderMain(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<Real> image, uint maxIterations, float maxSuperSampling, uint flags)
{
  const Pointi idx = getImageIndexes();
  if(idx.x >= outputSize.x || idx.y >= outputSize.y) return;
  if(idx.x == 0 && idx.y == 0){
    // printf("fractal render main\n");
    // printf("image:\t%f\t%f\t%f\t%f\n", image.left_bottom.x, image.left_bottom.y, image.right_top.x, image.right_top.y);
    // printf("outputsize:\t%d\t%d\n", outputSize.x, outputSize.y);
    // printf("focus:\t%d\t%d\n", focus.x, focus.y);
    // printf("dwell:\t%d\tss:\t%d\n", maxIterations, maxSuperSampling);
    // printf("\n");
  }
  
  //the value of maxSuperSampling will be changed by the calee
  uint result = sampleTheFractal(idx, outputSize, image, maxIterations, maxSuperSampling, flags & USE_ADAPTIVE_SS_FLAG_MASK);

  pixel_info_t* pOutput = getPtrToPixelInfo(output, outputPitch, idx);
  * pOutput = pixel_info_t(result, maxSuperSampling);
  //pOutput->value = 0;
  ASSERT(pOutput->weight > 0);
}







/// for given point <code>p</code> in the current image and given warping information, find cooridnates of the same point (=representing the same point in the fractal's complex plane) in the image being warped
/// @param p: the point whose warping origin is being returned (dimension: pixels)
/// @param gridSize: width and height (in pixels) of the grid (aka texture) we render on.
/// @param currentImage: rectangle representing the part of the complex plane that is being rendered (dimension: abs value in complex plane)
/// @param oldImage: rectangle representing the part of the complex plane that is being reused (dimension: abs value in complex plane)
template <class Real> __device__ __forceinline__
Point<float> getWarpingOriginOfSampleReuse(Point<uint> p, Point<uint> gridSize, Rectangle<Real> currentImage, Rectangle<Real> oldImage){
  
  p.y = gridSize.y - p.y; //switch y cordinate direction
  Point<Real> pixel_in_plane = p.cast<Real>() / gridSize.cast<Real>() * currentImage.size() +  currentImage.left_bottom;
  Point<Real> relative_in_old = (pixel_in_plane - oldImage.left_bottom) / oldImage.size(); //relative position of p with respect to oldImage
  Point<float> old_pixel = relative_in_old.cast<float>() * gridSize.cast<float>();
  old_pixel.y = gridSize.y - old_pixel.y; //switch y cordinate direction
  return old_pixel;
}

__device__ float screenDistance = 60; //in cm; better be set by the user
__device__ __forceinline__
/// Returns how many samples this pixel should take, based on foveation.
/// Value is between 0 and maxSuperSampling.
/// Value in the focus will always be maxSuperSampling, values in the non-peripheral view will always be non-zero.
/// Returned value is the same for all pixels within a warp (the highest is taken).
fov_result_t getFoveationAdvisedSampleCount(Pointi pixel, Pointi focus, float maxSuperSampling){
  //per-warp normalisation, i.e. set all pixels from a warp to same value
  pixel = pixel - (pixel % Pointi(WARP_SIZE_X, WARP_SIZE_Y));

  fov_result_t result;

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
  if(visualAngle <= fovealViewLimit){
    relativeQuality = 1;
    result.isInsideFocusArea = true;
  } 

  result.advisedSampleCount = maxSuperSampling * relativeQuality;
  // if(visualAngle <= peripheralViewTreshold)
  //   result = max(1.0, result); //always return at least 1 for pixels within the field of view
  return result;
}

template <class Real> __device__ 
pixel_info_t readFromArrayUsingLinearFiltering(pixel_info_t** textureArr, long textureArrPitch, Point<Real> coordinates){
 
    //implementation of bi linear filtering, see for example https://en.wikipedia.org/wiki/Bilinear_filtering

    Real xB = coordinates.x; //note that in canonical implementation, here is coordinates.x - 0.5. With our coordinate-system, we omit this.
    Real yB = coordinates.y; //ditto for y
    uint i = floor(xB);
    uint j = floor(yB);
    Real alpha = xB - i;
    Real beta = yB - j;

    pixel_info_t T_i_j = * getPtrToPixelInfo(textureArr, textureArrPitch, Point<uint>(i, j));
    pixel_info_t T_ip_j = * getPtrToPixelInfo(textureArr, textureArrPitch, Point<uint>(i+1, j));
    pixel_info_t T_i_jp = * getPtrToPixelInfo(textureArr, textureArrPitch, Point<uint>(i, j+1));
    pixel_info_t T_ip_jp = * getPtrToPixelInfo(textureArr, textureArrPitch, Point<uint>(i+1, j+1));

    //float weight_sum = T_i_j.weight + T_ip_j.weight + T_i_jp.weight + T_ip_jp.weight;

    //ASSERT(weight_sum != 0);
    float T_i_j_weighted = T_i_j.value ;//* T_i_j.weight / weight_sum;
    float T_ip_j_weighted  = T_ip_j.value ;//* T_ip_j.weight / weight_sum;
    float T_i_jp_weighted  = T_i_jp.value ;//* T_i_jp.weight / weight_sum;
    float T_ip_jp_weighted = T_ip_jp.value ;//* T_ip_jp.weight / weight_sum;


    Real r = (T_i_j_weighted   * (1-alpha)  + T_ip_j_weighted * alpha) * (1-beta) + 
            (T_i_jp_weighted * (1-alpha)  + T_ip_jp_weighted * alpha) * beta;
    float w = (T_i_j.weight   * (1-alpha)  + T_ip_j.weight * alpha) * (1-beta) + 
        (T_i_jp.weight * (1-alpha)  + T_ip_jp.weight * alpha) * beta;

    pixel_info_t result;
    result.value = r;
    result.weight = w;

    return result; 
}

template <class Real> __device__ 
void fractalRenderAdvanced(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<Real> image, uint maxIterations, float maxSuperSampling, uint flags, Rectangle<Real> imageReused, pixel_info_t** input, long inputPitch, Pointi focus){

  const Pointi idx = getImageIndexes();
  if(idx.x >= outputSize.x || idx.y >= outputSize.y) return;
  ASSERT(idx.x < outputSize.x);
  ASSERT(idx.y < outputSize.y);


  //foveation - only if zooming:
  fov_result_t fovResult(maxSuperSampling, false);
  if (flags & USE_FOVEATION_FLAG_MASK && flags & IS_ZOOMING_FLAG_MASK && maxSuperSampling >= 1){
    fovResult = getFoveationAdvisedSampleCount(idx, focus, maxSuperSampling);
  }
  // if(idx.x == focus.x && idx.y == focus.y){
  //  printf("%i\t%f\n",insideFocusArea, maxSuperSampling);
  // }
   
  // //at least one sample has to be taken somewhere
  // else if(advisedSampleCount == 0  && !reusingSamples ){
  //   advisedSampleCount = 1; 
  // }

  //sample reuse
  bool reusingSamples = false;
  pixel_info_t reused;
  if(flags & USE_SAMPLE_REUSE_FLAG_MASK){
    const Point<float> origin = getWarpingOriginOfSampleReuse(Point<uint>(idx.x, idx.y),outputSize,image, imageReused);
    const Point<int> origin_int = Point<int>(round(origin.x), round(origin.y));

    if(origin_int.x < 2 || origin_int.x >= outputSize.x - 2 || origin_int.y < 2 || origin_int.y >= outputSize.y - 2) {
      //if reusing would be out of bounds (i.e. no data to reuse)
      reusingSamples = false;
    }
    else if((flags & ZOOMING_IN_FLAG_MASK) && fovResult.isInsideFocusArea){
      //if zooming in and around the zooming center, there is little data to reuse
      reusingSamples = false;
    }
    else{
      reused = readFromArrayUsingLinearFiltering(input, inputPitch, origin);
      reusingSamples = true;
    }
  }


  float sampleCount = fovResult.advisedSampleCount; 
  pixel_info_t result;
  if(reusingSamples){
    result = reused;   
    result.isReused = true; //todo this is debug only
    // if(result.weight < 1)
      // result.value = 256* 0;
  }else{
    if(sampleCount < 1){
      sampleCount = 1; //at least one sample has to be taken somewhere
    }
    result.value = sampleTheFractal(idx, outputSize, image, maxIterations, sampleCount, flags & USE_ADAPTIVE_SS_FLAG_MASK);
    result.weight = sampleCount; //sampleCount is an in-out parameter
  }

  pixel_info_t* pOutput = getPtrToPixelInfo(output, outputPitch, idx);
  * pOutput = result;
  //ASSERT(result.weight > 0);
}



/**
 * TODO java-style documentation -- odkazuju se na to v textu baka
 *
 */
extern "C" __global__
void fractalRenderMainFloat(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<float> image, uint maxIterations, float maxSuperSampling, uint flags){
  fractalRenderMain<float>(output, outputPitch, outputSize, image, maxIterations, maxSuperSampling, flags);
}

extern "C" __global__
void fractalRenderMainDouble(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<double> image, uint maxIterations, float maxSuperSampling, uint flags){
  fractalRenderMain<double>(output, outputPitch, outputSize, image, maxIterations, maxSuperSampling, flags);

}

extern "C" __global__
void fractalRenderAdvancedFloat(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<float> image, uint maxIterations, float maxSuperSampling, uint flags, Rectangle<float> imageReused, pixel_info_t** input, long inputPitch, Pointi focus){
  fractalRenderAdvanced<float>(output, outputPitch, outputSize, image, maxIterations, maxSuperSampling, flags, imageReused, input, inputPitch, focus);
}

extern "C" __global__
void fractalRenderAdvancedDouble(pixel_info_t** output, long outputPitch, Pointi outputSize, Rectangle<double> image, uint maxIterations, float maxSuperSampling, uint flags, Rectangle<double> imageReused, pixel_info_t** input, long inputPitch, Pointi focus){
  fractalRenderAdvanced<double>(output, outputPitch, outputSize, image, maxIterations, maxSuperSampling, flags, imageReused, input, inputPitch, focus);

}

/// param maxSuperSampling: is here only for the purpose of visualizing the sample count
extern "C" __global__
void compose(pixel_info_t** inputMain, long inputMainPitch, pixel_info_t** inputBcg, long inputBcgPitch, cudaSurfaceObject_t  surfaceOutput, uint width, uint height, cudaSurfaceObject_t colorPalette, uint paletteLength, float maxSuperSampling){
    const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const Point<uint> idx(idx_x, idx_y);
    if(idx_x >= width || idx_y >= height) return;

    pixel_info_t result = * getPtrToPixelInfo(inputMain, inputMainPitch, idx);

    uint resultColor;
    if(VISUALIZE_SAMPLE_COUNT){ 
      resultColor = colorizeSampleCount(result.weight, maxSuperSampling);
      if(result.isReused){ //todo debug only
        resultColor = ColorsRGBA::BLACK;
      }
    }
    else{
      resultColor = colorize(colorPalette, paletteLength, result.value);
    }
    surf2Dwrite(resultColor, surfaceOutput, idx_x * sizeof(uint), idx_y);
}


extern "C" __global__
void fractalRenderUnderSampled(pixel_info_t** output, long outputPitch, uint width, uint height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, uint maxIterations, uint underSamplingLevel)
{
  //work only at every Nth pixel:
  const uint idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * underSamplingLevel;
  const uint idx_y = (blockDim.y * blockIdx.y + threadIdx.y) * underSamplingLevel;
  const Point<uint> idx(idx_x, idx_y);
  if(idx_x >= width-underSamplingLevel || idx_y >= height-underSamplingLevel) return;
  
  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float pixelWidth = (right_top_x - left_bottom_x) / (float) width;
  float pixelHeight = (right_top_y - left_bottom_y) / (float) height;
  
  float cx = left_bottom_x + (idx_x)  * pixelWidth;
  float cy = right_top_y - (idx_y) * pixelHeight;

  uint escapeTime = iterate(maxIterations, Pointf(cx, cy));

  for(uint x = 0; x < underSamplingLevel; x++){
    for(uint y = 0; y < underSamplingLevel; y++){
      pixel_info_t* pOutput = getPtrToPixelInfo(output, outputPitch, idx);
      * pOutput = pixel_info_t(escapeTime, 1 / (float) underSamplingLevel);
    }
  }

}

extern "C" __global__
void debug(){
  const uint idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x == 0 && idx_y == 0){
    // printf("aa:\t%u\n",a.a);
    debugFractal();
  }
}