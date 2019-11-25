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

/// Dispersion in this context is "Index of dispersion", aka variance-to-mean ratio. See https://en.wikipedia.org/wiki/Index_of_dispersion for more details
__device__
float computeDispersion(float* data, uint dataLength, float mean){
  uint n = dataLength;
  float variance = 0;
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
/// param image: segment of the complex plane that is to be rendered as an image (the region of interest).
/// param sampleCount: Maximum number of samples to take. Actual number of samples taken will be stored here before returning. The float value will always be round to the nearest int. If adaptiveSS==false, the value will not change (apart from rounding). 
template <class Real> __device__
uint sampleTheFractal(Pointi pixel, Pointi gridSize, Rectangle<Real> image, uint maxIterations,float & sampleCountF, bool adaptiveSS){
  if(sampleCountF < 1){
    sampleCountF = 0;
    return 0;
  }
  uint sampleCount = min(MAX_SUPER_SAMPLING, (uint) round(sampleCountF));

  const uint adaptiveTreshold = 10;
  float samples[adaptiveTreshold];

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  Point<Real> pixelSize = image.size() / gridSize.cast<Real>();
  
  uint result;
  uint escapeTimeSum = 0;
  ASSERT (sampleCount <= MAX_SUPER_SAMPLING);
  for(uint i = 0; i < sampleCount; i++){

    //todo tady bych radeji neco jako deltas[sampleCount]; createDeltas(& deltas) nebo tak. Zkratka abych nesamploval takhle primitivne po diagonale, ale lip. Random sampling na to samozrejme je nejlepsi, ale to asi fakt ma overhead? Nevim, jestli mam cas to merit.
    Point<Real> delta; //  = Point<Real>(i / (Real) sampleCount); //TODO TODO TODO PLS PLS PLS
    //todo jako druhou deltu zvolit pixelSize / 2.

    /// first {@code skipFirst} samples are taken diferently
    uint skipFirst = 2;
    if(i <= skipFirst){
      delta = Point<Real>(i / (Real) (skipFirst+1));
    } else {
      float samplesPerRowF = sqrt(sampleCountF-skipFirst);
      uint samplesPerRowI = round(samplesPerRowF);
      float dx = ((i-skipFirst) % samplesPerRowI) / samplesPerRowF;
      float dy = ((i-skipFirst) / samplesPerRowI) / samplesPerRowF;
      // if(dx > 1 || dy > 1){
      //   printf("dx: %f\tdy: %f\tSS: %f\tsqrt: %f\n", dx, dy, sampleCountF, samplesPerRowF);
      // }
      ASSERT(dy <= 1);
      ASSERT(dx <= 1);
      delta = Point<Real>(dx, dy);
    }
    
    const Point<Real> flipYAxis = Point<Real>(1,-1);
    const Point<Real> image_left_top = Point<Real>(image.left_bottom.x, image.right_top.y);

    /// a point in the complex plane that is to be rendered
    // c = {LT} {+,-} ((pixel+delta) * pixelSize)
    const Point<Real> c = image_left_top + flipYAxis * (pixel.cast<Real>() + delta) * pixelSize;

    uint escapeTime = computeFractal(maxIterations, c);
    escapeTimeSum += escapeTime;
    if(i < adaptiveTreshold){
      samples[i] = escapeTime;
    }

    //todo refactor
    if(adaptiveSS && ( i > 0 && i < adaptiveTreshold) || (i == sampleCount / 2) ){ //decide whether to continue with supersampling or not

      float mean = escapeTimeSum / (i+1);
      float dispersion = computeDispersion(samples, i, mean);       //todo toto do else vetve

      constexpr float treshold = 0.01;

      if(i == 1 && __ALL(abs(samples[0] - samples[1]) < FLT_EPSILON )){
        sampleCount = i+1; //terminating this cycle and storing info about actual number of samples taken
        //result = 255 * 0;  // blue  //this is for hypothesis testing only
      } 
      else if(__ALL(dispersion < treshold)){ // uniform distribution
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
  
  ASSERT(maxSuperSampling >= 1);
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

/// Apply the linear mapping given by (r,s) -> (a,b) on the x value
/// I.e. mapping p -> q, a -> b and any intermediate value lineary 
template <class Real> __device__ 
Real linearMapping(float x, float r, float s, float a, float b){
  Real k = (b-a)/(s-r);
  Real q = (s*a-b*r)/(s-r);
  return k*x+q;
}

__device__ float screenDistance = 60; //in cm; can be set by the user
__device__ float pixelRealWidthInCm = 0.02652; //in cm; can be set by the user; see http://www.prismo.ch/comparisons/desktop.php 
__device__
/// Returns how many samples this pixel should take, based on foveation.
/// Value is between 0 and maxSuperSampling, resp 0 or 1 for maxSuperSampling < 1.
/// Value in the focus will always be maxSuperSampling for maxSuperSampling >=1.
/// When maxSuperSampling < 1, then focus radius is lowered and pixels within still get result 1.
/// Returned value is the same for all pixels within a warp (the highest is taken).
fov_result_t getFoveationAdvisedSampleCount(const Pointi pixel, const Pointi focus, const float maxSuperSampling){
  ASSERT(maxSuperSampling > 0);

  //per-warp normalisation, i.e. set all pixels from a warp to same value
  const Pointi pixelN = pixel - (pixel % Pointi(WARP_SIZE_X, WARP_SIZE_Y));
  
  const float fovealViewTreshold = maxSuperSampling >= 1 ? 45.5 : 45.5 * maxSuperSampling; //in degrees
  const float peripheralViewTreshold = 60;  //in degrees

  fov_result_t result;

  float focusDistance = focus.cast<float>().distanceTo(pixelN.cast<float>()) * pixelRealWidthInCm; //distance to focus, translated to cm
  /// visual angle for one eye, i.e possible values are from 0 to ~ 110
  float visualAngle = atan (focusDistance / screenDistance) * 180 / PI_F; //from https://en.wikipedia.org/wiki/Visual_angle
  ASSERT(visualAngle >= 0); //arctan is positive for positive input and distance is always non-negative

  // //limited radius - this technique is not used in current release of chaos-ultra; it still needs some testing
  // const float consideredRadius = max(fovealViewTreshold,min(peripheralViewTreshold,linearMapping<float>(maxSuperSampling, 4, MAX_SUPER_SAMPLING, fovealViewTreshold, peripheralViewTreshold)));
  // if(visualAngle > consideredRadius)
  //   return fov_result_t(0,false); //for pixels after consideredRadious, no samples are desired

  //used model for (visualAngle -> relativeQuality): in (0,fovealViewTreshold): a constant function that yields 1, in (fovealViewTreshold, peripheralViewTreshold): descenidng linear function from 1 to 0
  float relativeQuality = min(1.0, linearMapping<float>(visualAngle, fovealViewTreshold, peripheralViewTreshold, 1, 0));
 
  result.advisedSampleCount = maxSuperSampling * relativeQuality;
  if(visualAngle <= fovealViewTreshold){
    result.advisedSampleCount = max(1.0, result.advisedSampleCount); //always return at least 1 for pixels within the foveal field of view
    result.isInsideFocusArea = true;
  } 
  return result;
}

template <class Real> __device__ 
pixel_info_t readFromArrayUsingFiltering(pixel_info_t** textureArr, long textureArrPitch, Point<Real> coordinates){
 
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

    float T_i_j_weighted = T_i_j.value ;
    float T_ip_j_weighted  = T_ip_j.value ;
    float T_i_jp_weighted  = T_i_jp.value ;
    float T_ip_jp_weighted = T_ip_jp.value ;

    //linear filtering:
    Real r = (1-alpha) * (1-beta) * T_i_j_weighted+
             alpha      * (1-beta) * T_ip_j_weighted+ 
             (1-alpha)  * beta     * T_i_jp_weighted +
             alpha      * beta     * T_ip_jp_weighted;
    float w = (1-alpha) * (1-beta) * T_i_j.weight +
              alpha      * (1-beta) * T_ip_j.weight + 
              (1-alpha)  * beta     * T_i_jp.weight +
              alpha      * beta     * T_ip_jp.weight;

    //or bilinear filtering:
    // Real r = (T_i_j_weighted   * (1-alpha)  + T_ip_j_weighted * alpha) * (1-beta) + 
    //         (T_i_jp_weighted * (1-alpha)  + T_ip_jp_weighted * alpha) * beta;
    // float w = (T_i_j.weight   * (1-alpha)  + T_ip_j.weight * alpha) * (1-beta) + 
    //     (T_i_jp.weight * (1-alpha)  + T_ip_jp.weight * alpha) * beta;



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


  //foveation - only if zooming in:
  fov_result_t fovResult(maxSuperSampling, false);
  if (flags & USE_FOVEATION_FLAG_MASK && flags & IS_ZOOMING_FLAG_MASK && flags & ZOOMING_IN_FLAG_MASK){
    fovResult = getFoveationAdvisedSampleCount(idx, focus, maxSuperSampling);
  }

  //sample reuse
  bool reusingSamples = false;
  pixel_info_t reused;
  if(flags & USE_SAMPLE_REUSE_FLAG_MASK){
    const Point<float> origin = getWarpingOriginOfSampleReuse(Point<uint>(idx.x, idx.y),outputSize,image, imageReused);
    const Point<int> origin_int = Point<int>(round(origin.x), round(origin.y));

    if(origin_int.x < 2 || origin_int.x >= outputSize.x - 2 || origin_int.y < 2 || origin_int.y >= outputSize.y - 2) {
      //if reusing would be out of bounds (i.e. no data to reuse)
      reusingSamples = false;
    }else{
      reused = readFromArrayUsingFiltering(input, inputPitch, origin);
      if(reused.weight < 0.1)
        reusingSamples = false;
      else
        reusingSamples = true;
    }
  }
  

  float sampleCount = fovResult.advisedSampleCount; 
  pixel_info_t result;
  if(reusingSamples){
    result = reused;   
    result.isReused = true; 
    result.weightOfNewSamples = 0;

    if((flags & ZOOMING_IN_FLAG_MASK) && (fovResult.isInsideFocusArea /*|| maxSuperSampling > 1 */)){
      //if around the zooming center during zooming in, the reuse data is innacurate -- we sample some more
      float samples = sampleTheFractal(idx, outputSize, image, maxIterations, sampleCount, flags & USE_ADAPTIVE_SS_FLAG_MASK);
      reused.weight *= 0.75;
      result.weightOfNewSamples = sampleCount;
      result.weight = reused.weight + sampleCount;
      result.value = (reused.value * reused.weight + samples * sampleCount) / result.weight;  
    }
  }else{
    if(sampleCount < 1){
      sampleCount = 1; //at least one sample has to be taken somewhere
    }
    result.value = sampleTheFractal(idx, outputSize, image, maxIterations, sampleCount, flags & USE_ADAPTIVE_SS_FLAG_MASK);
    result.weight = sampleCount; //sampleCount is an in-out parameter
  }

  pixel_info_t* pOutput = getPtrToPixelInfo(output, outputPitch, idx);
  * pOutput = result;
  ASSERT(result.weight > 0);
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
      resultColor = colorizeSampleCount(result.weight, max(1.0,maxSuperSampling));
      if(result.isReused){ 
        resultColor = colorizeSampleCount(result.weightOfNewSamples, max(1.0,maxSuperSampling));
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

  uint escapeTime = computeFractal(maxIterations, Pointf(cx, cy));

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