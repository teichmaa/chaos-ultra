#include <curand_kernel.h>
#include "math.h"
const int BLACK = 0xff000000;
const int WHITE = 0xffffffff;
const int PINK = 0xffb469ff;
const int YELLOW = 0xff00ffff;
const int GOLD = 0xff00d7ff;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Mandelbrot: Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)
  
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Mandelbrot: Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

//Mandelbrot content, using standard mathematical terminology for Mandelbrot set definition, i.e.
//  f_n = f_{n-1}^2 + c
//  f_0 = 0
//  thus iteratively applying: f(z) = z*z * c
//  where z, c are complex numbers, with components denoted as
//    x ... for real part (corresponding to geometric x-axis)
//    y ... for imag part (corresponding to geometric y-axis)

__device__ __forceinline__ int escape(int dwell, float cx, float cy){
  float zx = 0;
  float zy = 0;
  float zx_new;
  int i = 0;
  while(i < dwell && zx*zx+zy*zy < 4){
      zx_new = zx*zx-zy*zy + cx;
      zy = 2*zx*zy + cy; 
      zx = zx_new;
      ++i;
  }
  return i;
}

/// Dispersion in this context is "Index of dispersion", aka variance-to-mean ratio. See https://en.wikipedia.org/wiki/Index_of_dispersion for more details
__device__  __forceinline__ float computeDispersion(int* data, int dataLength, float mean){
  int n = dataLength;
  float variance = 0;
  for(int i = 0; i < dataLength; i++){
    //using numerically stable Two-Pass algorithm, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    variance += (data[i]-mean)*(data[i]-mean);
  }
  variance /= (n-1); 
  return variance / mean;
}


extern "C"
__global__ void mandelbrot(cudaSurfaceObject_t surfaceOutput, long outputDataPitch_debug, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, int dwell, int** outputData_debug, cudaSurfaceObject_t colorPalette, int paletteLength, float* randomSamples, int superSamplingLevel, bool adaptiveSS, bool visualiseSS)
// todo: usporadat poradi paramateru, cudaXXObjects predavat pointrem, ne kopirovanim (tohle rozmyslet, mozna je to takhle dobre)
//  todo ma to fakt hodne pointeru, mnoho z nich je pritom pro vsechny launche stejny - nezdrzuje tohle? omezene registry a tak
{
  const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x >= width || idx_y >= height) return;

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float pixelWidth = (right_top_x - left_bottom_x) / (float) width;
  float pixelHeight = (right_top_y - left_bottom_y) / (float) height;

  const int adaptiveTreshold = 10;
  int r[adaptiveTreshold];
  int adaptivnessUsed = 0;

  int escapeTimeSum = 0;
  //int randomSamplePixelsIdx = (idx_y * width + idx_x)*superSamplingLevel;
  for(int i = 0; i < superSamplingLevel; i++){
    //float random_xd = randomSamples[randomSamplePixelsIdx + i/2 ];
    //float random_yd = randomSamples[randomSamplePixelsIdx + i/2 + superSamplingLevel/2];
    float random_xd = i / (float) superSamplingLevel; //not really random, just uniform
    float random_yd = random_xd;
    float cx = left_bottom_x + (idx_x + random_xd)  * pixelWidth;
    float cy = right_top_y - (idx_y + random_yd) * pixelHeight;

    int escapeTime = escape(dwell, cx, cy);
    escapeTimeSum += escapeTime;
    if(i < adaptiveTreshold){
      r[i] = escapeTime;
    }
    if(i == adaptiveTreshold && adaptiveSS){ //decide whether to continue with supersampling or not
      float mean = escapeTimeSum / (i+1);
      float dispersion = computeDispersion(r, i, mean);
      if(dispersion <= 0.01){
        superSamplingLevel = i+1; //effectively disabling high SS and storing info about actual number of samples taken
        adaptivnessUsed = WHITE;
      }
      else if(dispersion <= 10){
        superSamplingLevel = min(i+1,superSamplingLevel / 2); //slightly reducing SS
        adaptivnessUsed = PINK;
      }
      else if(dispersion <= 100){
        superSamplingLevel = min(i+1,(int) (superSamplingLevel * 0.8f)); //slightly reducing SS
        adaptivnessUsed = GOLD;
      }else{ //else we are on an chaotic edge, thus as many samples as possible are needed
        adaptivnessUsed = BLACK;
      }
      
    }
  }
  int mean = escapeTimeSum / superSamplingLevel;  

  int paletteIdx = paletteLength - (mean % paletteLength) - 1;
  int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * 4, 0);
  if(mean == dwell)
    resultColor = BLACK;
  if(adaptivnessUsed && visualiseSS){
    resultColor = adaptivnessUsed;
  }
  /*if(idx_x < 10 && idx_y < 10){
    printf("%f\t", randomSample);
    __syncthreads();
    if(idx_x == 0 && idx_y == 0)
      printf("\n");
  }*/

  surf2Dwrite(resultColor, surfaceOutput, idx_x * sizeof(unsigned int), idx_y);
  //int* pOutputDebug = (int*)((char*)outputData_debug + idx_y * outputDataPitch_debug) + idx_x;
  //*pOutputDebug = result;

}

extern "C"
__global__ void init(){
  const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x == 0 && idx_y == 0){
    int sobolCount = 2 * 4096;
    unsigned int * sobolDirectionVectors = (unsigned int *) malloc(sobolCount * sizeof(unsigned int));
    curandStateSobol32_t * sobolStates= (curandStateSobol32_t *) malloc(sobolCount * sizeof(curandStateSobol32_t));
    printf("sizeof(curandStateSobol32_t): %d\n", sizeof(curandStateSobol32_t));
    if(sobolDirectionVectors == NULL || sobolStates == NULL){
      printf("init sobolDirectionVectors malloc failed");
      return;
    }
  }  
}