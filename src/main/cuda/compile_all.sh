#!/bin/sh
for f in test mandelbrot julia newton_wired newton_generic newton_iterations
do
    ./compile.sh $f
done