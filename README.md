# 3D-Hand-Tracking

This project is being developed under QMIND as a design team part of the DAIR division. **<a style="text-align:left" href="https://qmind.ca/#Research">
DAIR @ QMIND
</a>**

We aim to develop a Tensorflow model to predict the 3D shape and pose of two-hands through high hand-to-hand and hand-to-object contact, using just a monocular RGB input.
This project is still under development - for anything regarding this project, see the project <a href="/TODO.md">roadmap</a>.

# Preliminary Results

## Tensorflow implementation of MANO 

We have written routines in Tensorflow for taking a MANO template mesh through pertubations as well as linear blend
skinning. Pictured below is a rotated right hand rendered via open3D. In this image, the hand mesh is sampled as a pointcloud, and the joint locations are rendered as spheres.

<img width="413" alt="Screen Shot 2022-04-06 at 6 25 12 AM" src="https://user-images.githubusercontent.com/38915815/161954714-5e7b46cd-f3f9-445b-8329-2fefc9631994.png">

# Steps for Using

Simply clone the repo and run all code blocks in src/HandTracking.ipynb. Pay careful attention to any comments at the top of the code blocks, as some are only meant to run when using the project from within Google Colab.

## Paper

To build the paper via LaTeX, you will need to install an appropriate Tex distribution on your system. See https://www.overleaf.com/learn/latex/Choosing_a_LaTeX_Compiler.

Simply navigate to the paper subdirectory and run,

```
./build.sh
```



