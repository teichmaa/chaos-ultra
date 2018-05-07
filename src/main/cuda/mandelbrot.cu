#include <curand_kernel.h>
//#include "sobol_direction_vectors.h"
#include "math.h"
const int BLACK = 0xff000000;
const int WHITE = 0xffffffff;
const int PINK = 0xffff69b4;
const int YELLOW = 0xff00ffff;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
  printf("Mandelbrot content: Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)
  
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Mandelbrot content: Error at %s:%d\n",__FILE__,__LINE__); \
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

__device__  __forceinline__ float computeMean(int* data, int dataLength){
  int sum =0;
  for(int i = 0; i < dataLength; i++){
    sum += (data[i]);
  }
  return sum / dataLength;
}

__device__  __forceinline__ float computeDispersion(int* data, int dataLength, float mean){
  int n = dataLength;
  float variance = 0;
  for(int i = 0; i < dataLength; i++){
    variance += (data[i]-mean)*(data[i]-mean);
  }
  variance /= (n-1); 
  return variance / mean;
}


extern "C"
__global__ void mandelbrot(cudaSurfaceObject_t surfaceOutput, long outputDataPitch_debug,/*int * palette,*/int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, int dwell, int** outputData_debug, cudaSurfaceObject_t colorPalette, int paletteLength, float* randomSamples, int superSamplingLevel)
// todo: usporadat poradi paramateru, cudaXXObjects predavat pointrem, ne kopirovanim (tohle rozmyslet, mozna je to takhle dobre)
//  todo na paletu by byla rychlejsi textura nez surface, ale to mi nefungovalo (vracelo jen dolnich 8 bytes)
//  todo ma to fakt hodne pointeru, mnoho z nich je pritom pro vsechny launche stejny - nezdrzuje tohle? omezene registry a tak
{
  const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x >= width || idx_y >= height) return;

  // if(idx_x == 0 && idx_y == 0)
  //     printf("dwell: %d\n", dwell);

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float pixelWidth = (right_top_x - left_bottom_x) / (float) width;
  float pixelHeight = (right_top_y - left_bottom_y) / (float) height;

  const int adaptiveTreshold = 3;
  int r[adaptiveTreshold];
  int adaptivnessUsed = 0;

  int escapeTimeSum = 0;
  int randomSamplePixelsIdx = (idx_y * width + idx_x)*superSamplingLevel;
  for(int i = 0; i < superSamplingLevel; i++){
    float random_xd = randomSamples[randomSamplePixelsIdx + i/2 ];
    float random_yd = randomSamples[randomSamplePixelsIdx + i/2 + superSamplingLevel/2];
    float cx = left_bottom_x + (idx_x + random_xd)  * pixelWidth;
    float cy = right_top_y - (idx_y + random_yd) * pixelHeight;

    int escapeTime = escape(dwell, cx, cy);
    escapeTimeSum += escapeTime;
    if(i < adaptiveTreshold){
      r[i] = escapeTime;
    }
    if(i == adaptiveTreshold){ //decide whether to continue with supersampling or not
      float mean = escapeTimeSum / (i+1);
      float dispersion = computeDispersion(r, i, mean);
      if(dispersion <= 0.1){
        superSamplingLevel = i+1; //effectively disabling high SS and storing info about actual number of samples taken
        //adaptivnessUsed = WHITE;
      }
      else if(dispersion <= 10){
        superSamplingLevel = min(i*3,superSamplingLevel); //massively reducing SS, if possible
        //adaptivnessUsed = PINK;
      }
      else if(dispersion <= 100){
        superSamplingLevel = min(i+1,superSamplingLevel / 2); //slightly reducing SS
        //adaptivnessUsed = BLACK;
      }
      //else we are on an chaotic edge, thus as many samples as possible are needed
    }
  }
  int mean = escapeTimeSum / superSamplingLevel;  

  int paletteIdx = paletteLength - (mean % paletteLength) - 1;
  int resultColor;
  surf2Dread(&resultColor, colorPalette, paletteIdx * 4, 0);
  if(mean == dwell)
    resultColor = BLACK;
  /*if(adaptivnessUsed){
    resultColor = adaptivnessUsed;
  }*/
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

  /*
  unsigned int * direction_vectors = (unsigned int *) malloc(32 * sizeof(unsigned int));
  direction_vectors[0] = 1;
  if(direction_vectors == NULL){
      printf("thread (%d, %d) malloc failed", idx_x, idx_y);
      return;
  }
  
  curandStateSobol32_t sobolState;
  curand_init((unsigned int *) direction_vectors,(unsigned int) 0, & sobolState);
  for (int i = 0; i < 10; i ++){
    float rand = curand_uniform(&sobolState);
    if(idx_x < 10 && idx_y < 10){
      printf("thread (%d, %d) rand: %f\t", idx_x, idx_y, rand);
      __syncthreads();
      if(idx_x == 0 && idx_y == 0)
        printf("\n");
    }
  }
  
  free(direction_vectors);*/
  
}