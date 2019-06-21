#!/bin/sh
# usage: .\compile.sh <fractalname> [<nvcc args>]
# example: \compile.sh mandelbrot
#      second argument is optional
#          add to see all warnings: --compiler-options -Wall             
#
# ! corresponding <fractalname>.cu file must be contained in the fractals directory,
#      and the file "fractalRendererGeneric.cu" in the root directory

filename=tmp_compiling_$1.cu
dir=fractals
echo '#include' $dir/$1.cu > $filename
echo '#include fractalRendererGeneric.cu' >> $filename
echo nvcc -ptx $filename -o $dir/$1.ptx -arch=sm_30  $2 | sh -x
rm $filename