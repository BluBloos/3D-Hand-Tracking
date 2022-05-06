'''

Things done:
___________________________________

- Revist iterative regression module. as we saw odd things in the pytorch codebase.
    - We found that we are not supposed to have reLu activation at the output -> fixed.
    - Changed back to three iters.

- Default back to same hyperparams as seen in MobileHand (for loss terms).
    - Of course this is with our own addition of the division by 2.

- First test results of gcloud training.

- Decreasing the drop prob to about 0.4 -> decrease regularization to prevent underfitting.


Next steps that we believe are going to create a good impact for us.
___________________________________


- Put rotation of hand root back into MANO model. 
- Use intermediate supervision of camera parameters (scale).
    - And remove the z_depth from RHD images (because this is like, no!)

- Add decay to the learning rate when the loss plateaus via a callback function.
    - Can do via a scheduler function -> although I am not sure about precisely what the step variable is.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay

Previous Observations:
___________________________________

    -> Mode collapse on model pose.
    -> Rotations for z-axis appear sane.


Notes:
___________________________________

- What is L2 loss? L2 = sum: (y - y_pred)^2.

- What is bias and variance?
    - Bias is the average error between the estimation and the ground truth (miss the target, accuracy).
    - Variance is like, how difference are the estimates?? (precision).


Underfitting Research:
___________________________________
https://www.ibm.com/cloud/learn/underfitting#toc-how-to-avo-OLugiNSf

- Generally, underfitting means that you model cannot estimate for new data OR training data.
    - Characterized by high bias and low variance.
    - Reasons?
        - Lack of complexity.
        - Undertraining.

        (My own ideas):
        - Loss functions not proper.
        - Large domain gap.

How to fix:
- Decrease regularization.
- Train more.
- Add more features:
    - Ex: You might add a hidden layer -> more complexity.


As seen in Zhang et al.
___________________________________

- Consider decaying the learning rate when the loss plateaus -> we can use a callback function.
- Consider decreasing the drop prob to about 0.4 -> decrease regularization to prevent underfitting.
- They do centering of the hand, but then perform data augmentation during training.
- Add a prior layer before iterative regression for estimation of 2D heatmaps.
- Iterative regression module is the same, but MobileNetV3 replaced with from-scratch encoder.
    - Important difference is that the input of this layer is gon be the heatmaps!
- Add a differentiable render for supervision of segmentation mask -> Inverse Graphics technique.
- Put rotation of hand root back into MANO model. 
- Use intermediate supervision of camera parameters + 2D heatmap layers:
    - To ensure the deep layers are giving good output.

Geometric constraints:
- Tip to palm for any finger (except thumb) are in the same plane.



More Brainstorm:
___________________________________

- Generally skeptical of 2D / 3D loss.
    - We give the scale + z_depth to the model during 2D_LOSS function.
        -> this does not seem correct as we are providing something in the loss
        that was not a direct output of the model. This seems like we are breaking
        a golden rule. 
    - The 3D loss is depeneing on cam_R.


- Want to further review the iterative regression module AS:
    - Seeing that in the MobileHand Github -> when in non-evaluation mode, they output all three params???
        - does this have implications in the way we compute the loss for this layer??
    
- We could investigate deeply into MobileNetV3-Small
    - Explore unfreezing MobileNetV3-Small, but still loading pre-trained weights.
    - Consider visualizing the output features of the MobileNetV3 encoder (but can they even be visualized).
    - Confirm that we have the right cutoff point in MobileNetV3
       - Paper says, "Used up to average pooling layer to output 576 feature vectors."



Things that exist, but prob will not help (because the prob is underfitting):
___________________________________

    - We could generally think about weight initialization.
    - Think about the way we train the model
        - Learning rate
        - Training on STB.


'''