//const int ARGB_FULL_OPACITY_MASK = 0xff000000;
//const int RGBA_FULL_OPACITY_MASK = 0x000000ff;
const int BLACK = 0xff000000;

//Mandelbrot kernel, using standard mathematical terminology for Mandelbrot set definition, i.e.
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

extern "C"
__global__ void mandelbrot(cudaSurfaceObject_t surfaceOutput, long outputDataPitch_debug,/*int * palette,*/int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, int dwell, int** outputData_debug, cudaSurfaceObject_t colorPalette, int paletteLength)
// todo: usporadat poradi paramateru, cudaXXObjects predavat pointrem, ne kopirovanim (tohle rozmyslet, mozna je to takhle dobre)
//  todo na paletu by byla rychlejsi textura nez surface, ale to mi nefungovalo (vracelo jen dolnich 8 bytes)
{
  const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x >= width || idx_y >= height) return;

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float cx = left_bottom_x + idx_x / (float) width * (right_top_x - left_bottom_x);
  float cy = right_top_y - idx_y / (float) height * (right_top_y - left_bottom_y);

  int escapeTime = escape(dwell, cx, cy);

  int paletteIdx = paletteLength - (escapeTime % paletteLength) - 1;
  int result;
  surf2Dread(&result, colorPalette, paletteIdx * 4, 0);
  if(escapeTime == dwell)
    result = BLACK;

  surf2Dwrite(result, surfaceOutput, idx_x * sizeof(unsigned int), idx_y);
  //int* pOutputDebug = (int*)((char*)outputData_debug + idx_y * outputDataPitch_debug) + idx_x;
  //*pOutputDebug = result;

  /*
  //debug part:
  if(idx_x==0 && idx_y ==0){
    //* pResult = dwell;
  }
  if(idx_x==1 && idx_y ==0){
    * pResult = width;
  }
  if(idx_x==2 && idx_y ==0){
    * pResult = height;
  }
  if(idx_x==3 && idx_y ==0){
    * pResult = 42;
  }
  */
}