'''

Observations:
___________________________________

    -> Mode collapse on model pose.
    -> Rotations for z-axis appear sane.


Things we might want to do:
___________________________________

- Write a script for easily download checkpoints from the Google bucket.

- Revist the iterative regression module as we saw odd things in the pytorch codebase.
    - Did the revisit -> figured out that we are not supposed to have activation at the output
    of our iterative regressor -> I feel like this was HUGE.
    - Also changed back to three iters.
    - Also seeing that in the MobileHand Github -> when in non-evaluation mode, they output all three params???
        - does this have implications in the way we compute the loss for this layer??
        -> for now we will put this aside.

- Something we could try is to simply put the rotation of the root back into the model.
Nothing odd is happening now with the 3D loss. extrinsic camera simply applies translation and rotation.
    -> also a great question is why we don't just take the RHD images and give them a depth of zero.
    -> because we provide this info to our model anyways -> so just be consistent.

- We could explore unfreezing but still loading a pre-trained MobileNextV3. So weights get modified.
    -> Hasson et al is saying that they froze the batch norm layers?

- Considering using the same loss coefficients as seen in Mobilehand.
    - Will need to of course consider that we use the full 45 MANO pose parameters.

- Generally skeptical of that our 3D keypoint loss depends on the cam_R being estimated proper :(
    - All we would need to do is the following 
        -> generate our own ground truth cam_R.
        -> use this ground truth to apply inverse camera_extrinsic to 
        GT 3D keypoints for comparison with output from MANO.
            -> the the 3D keypoint output is prior to translation, scaling, and rotation.
            -> because we want to estimate a prior.

- Unfreeze MobileNetV3
- Maybe read online about underfitting. Do our research and see if we can learn something fundamental 
    or new about machine learning.
- Confirm that we have the right cutoff point in MobileNetV3
    - Paper says, "Used up to average pooling layer to output 576 feature vectors."
- Think about weight initialization.
- Think about the way we train the model
    - Learning rate
    - Training on STB.
- Consider visualizing the output features of the MobileNetV3 encoder.


'''