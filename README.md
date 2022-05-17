# 3D-Hand-Tracking

This project was developed under QMIND as a design team part of the DAIR division. **<a style="text-align:left" href="https://qmind.ca/#Research">
DAIR @ QMIND
</a>**

We built a Tensorflow model to predict the 3D shape and pose of a hand using just a monocular RGB input. Our ultimate goal is to extend the model to predict two-hands through high hand-to-hand and hand-to-object contact.

# Results

Our model is capable of achieving an AUC score of 0.940 on the RHD dataset when evaluating on the training split. This of course means that our model overfits and does not generalize well to unseen examples.

Pictured below are two figures that the team included in our paper submitted to the CUCAI 2022 proceedings. For our full paper, you can find this as a pdf in the results sudirectory of this repo.

<img width="500" alt="Picture2" src="https://user-images.githubusercontent.com/38915815/168714866-d535b2d0-ffe0-4e8b-8bbc-ddf34e810aff.png"> <img width="500" alt="Picture2" src="https://user-images.githubusercontent.com/38915815/168714691-737a8959-7437-4cbb-83c8-6a9a0667cbaf.png">

The image on the top is the 3D PCK curve, which is where the AUC score comes from (area under the curve). The 3D PCK is the percentage of estimated hand keypoints that are within a specific error threshold to the ground truth keypoints, and the 3D PCK curve takes this metric across different error thresholds. 

The image on the bottom is a composite image of sample images chosen from the training split of the RHD dataset, and the associated predictions from our trained model.

## Tensorflow implementation of Linear Blend Skinning 

Something that the team is proud of is our custom implementation of Linear Blend Skinning written in Tensorflow. The code takes a MANO template mesh through the rotation parameters estimated by our model. Pictured below is a rotated right hand rendered via open3D. In this image, the hand mesh is sampled as a pointcloud, and the joint locations are rendered as spheres.

<img width="413" alt="Screen Shot 2022-04-06 at 6 25 12 AM" src="https://user-images.githubusercontent.com/38915815/161954714-5e7b46cd-f3f9-445b-8329-2fefc9631994.png">

# Steps for Using

Instructions for interacting with our codebase are currently "in the works" due to many recent, rapid changes.



