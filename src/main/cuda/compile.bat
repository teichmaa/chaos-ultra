:: usage: compile.bat <fractalname> [<nvcc args>]
:: example: compil.bate mandelbrot
::      second argument is optional
::
:: ! corresponding <fractalname>.cu file must be contained in the directory,
::      same as the "fractalRendererGeneric.cu" file

@set filename=tmp_compiling_%1.cu
@(
@    echo #include "%1.cu"
@    echo #include "fractalRendererGeneric.cu"
 ) > %filename%
nvcc -ptx %filename% -o %1.ptx -arch=sm_30  %2
@del %filename% /q