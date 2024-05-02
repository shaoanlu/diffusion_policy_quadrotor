## Learning note
### Main insight
- The model should learn the residuals (gradient of the denoising) if possible. This greatly stabilizes the training.
- Advantages of diffusion model: 1) capability of modeling multi-modality, 2) stable training, and 3) temporally output consistency.
- Iteratively add training data of failure modes to make extrapolation into interpolation.
### Scribbles
- The trained policy does not 100% reach the goal without collision (there is no collision in its training data).
  - Unable to recover from OOD data.
- Long observation might be harmful to the performance, possibly due to the increased possibility of model overfitting.
  - The diffusion policy [paper](https://arxiv.org/pdf/2303.04137) also discusses this in its appendix.
- I feel we don't need diffusion models for the simple task in this repo, supervised learning might be equally effective.
- The controller struggles with performance issues when extreme values (maximum or minimum) are presented in the conditional vector.
  - For instance, it collides more on obstacles with a maximum radius of 1.5.
  - Collect more data to make everything interpolations instead of extrapolations.
- Even though the loss curve appears saturated, the performance of the controller can still improve as training continues.
- The training loss curves of the diffusion model are extremely smooth btw.
  - On the contrary, it might be difficult to know if the model is overfitting or not by looking at the trajectory as well as the the denoising process.
  - But in general I feel there is little harm training duffusion model as long as possible.
- DDPM and DDIM samplers yield the best result.
- Inference is not in real-time. The controller is set to sun 100Hz.

### Possible reasons for failures on collision avoidance
1. There is no data having collision in the training data. 
2. Policy learned with imitation learning can exhibit accumulated error during closed-loop control

When the quadrotor getting too close to the obstacles (due to 2), the input state becomes OOD (due to 1), therefore the diffusion policy is unable to recover from such situation.

- Possible fix: adding training data that the quadrotor recovers from collision.

### Things that didn't work
- Tried encoding distances to each obstacle. Did not observe improvement in terms of collision avoidance.
- Tried using vision encoder to replace obstacle encoding. Didn't see performance improvement.