# 3D-Hand-Tracking

This project is being developed under QMIND as a design team part of the DAIR division. **<a style="text-align:left" href="https://qmind.ca/#Research">
DAIR @ QMIND
</a>**

We aim to develop a Tensorflow model to predict the 3D shape and pose of two-hands through high hand-to-hand and hand-to-object contact, using just a monocular RGB input.
This project is still under development - for anything regarding this project, see the project <a href="/TODO.md">roadmap</a>. 

# Preliminary Results

Below we present our preliminary results, that being an implementation of UNET to predict the segmentation mask of images from the <a href="https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html">RHD dataset</a>.

Pictured from Left -> Right

Input image, Ground truth segmentation mask, Model prediction

<div style="display:flex; flex-direction:row;">
<img width="264" alt="Screen Shot 2021-12-08 at 4 51 07 PM" src="https://user-images.githubusercontent.com/38915815/145290279-1e4a2250-e7be-48fc-b3dc-30ecf2a63d03.png">
<img width="264" alt="Screen Shot 2021-12-08 at 4 51 15 PM" src="https://user-images.githubusercontent.com/38915815/145290290-48eac1cf-21da-481c-bd0d-6c582623b976.png">
<img width="264" alt="Screen Shot 2021-12-08 at 4 51 33 PM" src="https://user-images.githubusercontent.com/38915815/145290292-f546ce0f-7178-49d0-9504-8d227f0ebacc.png">
</div>

# How to Use

Simply clone the repo and run all code blocks in src/HandTracking.ipynb. Pay careful attention to any comments at the top of the code blocks, as some are only meant to run when using the project from within Google Colab. 



