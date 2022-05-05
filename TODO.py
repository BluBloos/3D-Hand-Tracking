'''

Observations:
___________________________________

    -> Mode collapse on model pose.
    -> Rotations for z-axis appear sane.


Things we might want to do:
___________________________________

- Write a script for easily download checkpoints from the Google bucket.

- Revist the iterative regression modile as we saw odd things in the pytorch codebase.
    - I'm generally skeptical of this whole iterative reg module and how we use the same weights for the three iterations?

- Considering using the same loss coefficients as seen in Mobilehand.
    - Will need to of course consider that we use the full 45 MANO pose parameters.

- Unfreeze MobileNetV3
- Confirm that we have the right cutoff point in MobileNetV3
    - Paper says, "Used up to average pooling layer to output 576 feature vectors."
- Think about weight initialization.
- Think about the way we train the model
    - Learning rate
    - Training on STB.
- Consider visualizing the output features of the MobileNetV3 encoder.


'''