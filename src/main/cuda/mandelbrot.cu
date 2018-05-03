const int ARGB_FULL_OPACITY_MASK = 0xff000000;

//const int RGBA_FULL_OPACITY_MASK = 0x000000ff;

extern "C"
__global__ void mandelbrot(cudaSurfaceObject_t surface, long pitch,/*int * palette,*/int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, int dwell, int** outputData)
{
    const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(idx_x >= width || idx_y >= height) return;
    
    bool tooNear = false;
    const float FLOAT_MIN = 1E-5;
    const float FLOAT_MIN_NEG = -FLOAT_MIN;
    if( (FLOAT_MIN_NEG < left_bottom_x && left_bottom_x < FLOAT_MIN) ||
        (FLOAT_MIN_NEG < left_bottom_y && left_bottom_y < FLOAT_MIN) ||
        (FLOAT_MIN_NEG < right_top_x && right_top_x < FLOAT_MIN) ||
        (FLOAT_MIN_NEG < right_top_y && right_top_y < FLOAT_MIN)
    ){
      tooNear = true;
    }

    //napad: neco jako if too near, pocitej v double
    //  ale mel by to byt rovnou launcher - ze spusti alternativni kernel
    //  neco jako if(Kernel.hasSuportHQ) Kernel.getMainFunctionHQ
    // a taky bych si tyhle poznamky mel psat na jedno centralni misto

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
  //  int* pResult = (int*)((char*)outputData + idx_y * pitch) + idx_x;
    //* pResult = i | ARGB_FULL_OPACITY_MASK;
    //*pResult = result;
    unsigned int result = i << 16 | ARGB_FULL_OPACITY_MASK;

    if(tooNear)
      result = 0xff0000ff;

    surf2Dwrite(result, surface, idx_x * sizeof(unsigned int), idx_y);

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