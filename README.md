# chaos-ultra
Real time high quality zoomer of complex chaotic functions (fractals), e.g. the Mandelbrot set.

## WARNING
This project is work in progress. No release is available yet. The code may contain (and it does) hacky solutions that are meant as temporal and to be fixed in the release version.

This project originated as a bachelor thesis on [MFF UK](https://mff.cuni.cz/).

## Running the app
Java parameters 
 * param colorPalette: use `-DcolorPalette={filePath}` java argument to load a color palette from custom location. Default location is `palette.png`. If you don't specify the parameter, palette is searched in the default location. If no palette is found, a software-default palette is used.
 
 * param cudaKernelsDir: use `-DcudaKernelsDir={dirRelPath}` java argument to specify where your compiled cuda modules (.ptx files) are stored. Default is `cudaKernels`. If you don't specify this argument, the default value is used and a warning is given.
 
 * param renderingLogging: use `-DcudaKernelsDir=true` to enable logging of rendering information, for example frame render time. Default value is `false`. Note: The logs are printed in a blocking manner, after every frame, and thus enabling the logging may introduce performance overhead.

## Author

**Antonín Teichmann** - [teichmaa](https://github.com/teichmaa)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
