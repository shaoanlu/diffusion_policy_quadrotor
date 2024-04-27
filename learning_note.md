## Learning note
### Main insight
- The model learns the residuals (velocity, gradient of the denoising) if possible. This greatly stabilizes the training.
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
- DDPM and DDIM samplers yield the best.
- Inference is not in real-time. The controller is set to sun 100Hz.

### Things that didn't work
- Tried encoding distances to each obstacle. Did not observe improvement in terms of collision avoidance.