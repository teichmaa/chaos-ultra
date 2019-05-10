:: usage: compile.bat <fractalname> [<nvcc args>]
:: example: compile.bat mandelbrot
::      second argument is optional
::          add to see all warnings: --compiler-options -Wall             
::
:: ! corresponding <fractalname>.cu file must be contained in the fractals directory,
::      and the file "fractalRendererGeneric.cu" in the root directory


@set filename=tmp_compiling_%1.cu
@set dir=fractals
@(
@    echo #include "%dir%/%1.cu"
@    echo #include "fractalRendererGeneric.cu"
 ) > %filename%
nvcc -ptx %filename% -o %dir%/%1.ptx -arch=sm_30  %2  
@del %filename% /q