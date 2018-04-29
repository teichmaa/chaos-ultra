const int ARGB_FULL_OPACITY_MASK = 0xff000000;

extern "C"
__global__ void mandelbrot(int** outputData, long pitch,/*int * palette,*/ int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, int dwell)
{
  const unsigned  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(idx_x >= width || idx_y >= height) return;

  float zx = 0;
  float zy = 0;
  float zx_new;

  //We are in a complex plane from (left_bottom) to (right_top), so we scale the pixels to it
  float cx = left_bottom_x + idx_x / (float) width * (right_top_x - left_bottom_x);
  float cy = right_top_y - idx_y / (float) height * (right_top_y - left_bottom_y);


  int i = 0;
  while(i < dwell && zx*zx+zy*zy < 4){
      zx_new = zx*zx-zy*zy + cx;
      zy = 2*zx*zy + cy; 
      zx = zx_new;
      ++i;
  }
   int* pResult = (int*)((char*)outputData + idx_y * pitch) + idx_x;
  * pResult = i | ARGB_FULL_OPACITY_MASK;
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

/*
CUDA snippets

The following snippets are available :

__s : __syncthreads();
cmal : cudaMalloc((void**)&${1:variable}, ${2:bytes});
cmalmng : cudaMallocManaged((void**)&${1:variable}, ${2:bytes});
cmem : cudaMemcpy(${1:dest}, ${2:src}, ${3:bytes}, cudaMemcpy${4:Host}To${5:Device});
cfree : cudaFree(${1:variable});
kerneldef : __global__ void ${1:kernel}(${2}) {\n}
kernelcall : ${1:kernel}<<<${2},${3}>>>(${4});
thrusthv : thrust::host_vector<${1:char}> v$0;
thrustdv : thrust::device_vector<${1:char}> v$0;

*/