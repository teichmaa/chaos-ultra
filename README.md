# chaos-ultra
Real time high quality renderer and zoomer of complex chaotic functions, especially Escape-time fractals, e.g. the Mandelbrot set.

The program has been inspired by [Xaos](http://matek.hu/xaos/doku.php), a real-time fractal zoomer, and [ultrafractal](https://www.ultrafractal.com/), a high quality fractal renderer. Its backend runs on the GPU, currently only CUDA is supported.

## About

This program arose as the software-part of a bachelor thesis in 2018/2019. The corresponging thesis is *Real-time visualization of chaotic functions* by Antonín Teichmann at Charles University, Prague. Read the thesis to learn more details about source code structure, the logic of the program, and about rendering of chaotic functions in general.

## Technical requirements

To run the program, CUDA-capable GPU by nvidia is needed. A device with CUDA version 6 or higher is needed (we target compute capability 3.0) Check [this link](https://www.geforce.com/hardware/technology/cuda/supported-gpus) to see if you have a CUDA-capable GPU. A mid-end GPU, for example GeForce GTX 1060, is expected for smooth real-time performance.

### Software prerequisites

* Operating system: Linux and Windows are officially supported. The program is written in Java and should run on every Java- and CUDA-capable system.
* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
  * This includes the `nvcc` compiler.
* Java 8 or higher, including JavaFX. JavaFX is standard part of Oracle Java; if you are using `openjdk`, you will need to install `openjfx`.
* maven, to build the Java program.
* A c++ compiler, to build the CUDA-backend.

## Build and compilation

IDE: The program has been build using IntelliJ Idea (for Java) and VSCode (for cuda-backed). Arbitrary Java and CUDA IDE could be used.

### Java

We use maven. To compile the application, run `mvn compile`. To create an executable jar, run `mvn package`. 

There are no special build steps; in theory, other java build processes could be used too.

### cuda-backend

The cuda-backend is written in CUDA C/C++, a C++-based programming language introduced by nvidia.

For compiling source code in C/C++, the nvcc compiler by nvidia is needed. nvcc relies on a regular C++ compiler, which needs to be installed too.

For building the program, we use a custom script. The script is named `compile`, located in `src/main/cuda/`, and its usage is following:

* Prerequisite: A fractal named `{filename}.cu` must be located in `fractals/`
* run `compile.bat {filename} {options}` on Windows
* run `compile.sh {filename} {options}` on Linux
* `{options}` are passed to the c++ compiler and may be left blank
* For example, to compile `fractals/mandelbrot.cu`, use: `compile.bat mandelbrot`
* This produces `{filename}.ptx`

To compile all available fractals, update the fractal list in `compile_all{.bat , .sh}` and run the script.  

## Download

If you do not want to compile the program yourself, you can download latest executable at [here](https://gimli.ms.mff.cuni.cz/~tonik/chaos-ultra.zip). 


## Running the program

Start the program with `java DcudaKernelsDir=../src/main/cuda/fractals -jar chaos-ultra-1.0-jar-with-dependencies.jar`.

Without the `cudaKernelsDir` parameter specified, the `cuda_kernels` directory must be located in the same folder as the jar, and it must contain .ptx files with compiled fractal implementation for all the fractals registered in `CudaFractalRendererProvider`.


To modify program's behavior, following Java arguments can be used:

 * param colorPalette: use `-DcolorPalette={filePath}` java argument to load a color palette from custom location. Default location is `palette.png`. If you don't specify the parameter, palette is searched in the default location. If no palette is found, a software-default palette is used.
 
 * param cudaKernelsDir: use `-DcudaKernelsDir={dirRelPath}` java argument to specify where your compiled cuda modules (.ptx files) are stored. Default is `cudaKernels`. If you don't specify this argument, the default value is used and a warning is given.
 
 * param renderingLogging: use `-DrenderingLogging=true` to enable logging of rendering information, for example frame render time. Default value is `false`. Note: The logs are printed in a blocking manner, after every frame, and thus enabling the logging may introduce performance overhead.
 
 * param debug: use `-Ddebug=true` to enable debug output. Default value is `false`.
 
 ### Invalid ptx error
 
 If you get the `CUDA_ERROR_INVALID_PTX` when launching the program with the custom fractals, you are probably using a CUDA device with CUDA-version 5 or lower, with no support for compute capability 3.0.
 
 Such devices are not supported. However, you can try to modify the compilation script, lowering the target compute capability. Then, **maybe**, chaos-ultra could work with an unsupported GPU.
 
## Developer, the GUI and fractal lifecycle

In the GUI in the *additional functionality* pane, there are developer and debugging functions.

* *Invoke debug method* button invokes the `debugFractal` method of currently selected fractal on the device in a 1×1 grid, i.e. the debug method will be called exactly once.
* *reload* button reloads the implementation of current fractal from its .ptx file. This is useful especially during development, to reload the fractal after it has been recompiled, without the need to restart the application.

When the fractal is changed in the GUI, corresponding module is freed and closed, and the new module is loaded from its .ptx file. After the module has been loaded, changes in its .ptx source file do not affect it until it is reloaded by clicking the *reload* button or until the fractal has been changed to some other and then back.



## Troubleshooting

Errors are printed in the GUI, in *system messages* area, but often, the error message is not sufficient without exception name or the stack trace.

We recommend checking the error output of the program and inspect the stack trace. The exception names and messages of the program are usually expressive and telling, hence an experienced programmer should be fine. 

Errors that happened on the device are hard to debug. We recommend reading more about it [here](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm).

If you find a bug, I will be more than happy if you contact me, so that I can fix it, or if you make a pull request. 

## Author

**Antonín Teichmann** - [teichmaa](https://github.com/teichmaa)

## Versioning

Version number is in format `major.minor`, `major` being number of major version, `minor` number of minor version. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
