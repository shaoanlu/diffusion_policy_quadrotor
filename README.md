# diffusion_policy_quadrotor
This repository provides a demonstration of imitation learning using a diffusion policy. The implementation is adapted from the official Diffusion Policy [repository](https://github.com/real-stanford/diffusion_policy).

## Result
The control task is to drive the quadrotor from the initial position (0, 0) to the goal position (5, 5) without collision with the obstacles. The animation shows the denoising process of the diffusion policy predicting future trajectory followed by the quadrotor applying the actions. 

<img src="assets/result_anim.gif" alt="drawing" width="380"/> <img src="assets/result_plot.png" alt="drawing" width="370"/>


## Usage
The notebook `demo.ipynb` demonstrates a closed-loop simulation using the diffusion policy controller for quadrotor collision avoidance.

## Dependencies
The program was developed and tested in the following environment.
- Python 3.10
- `torch==0.13.1`
- `jax==0.4.23`
- `jaxlib==0.4.23`
- `diffusers==0.18.2`
- `torchvision==0.14.1`
- `gdown` (to download pre-trained weights)
- `joblib` (format of training data)

## Diffusion policy
The policy takes 1) the latest N step of observation $o_t$ (position and velocity) and 2) the encoding of obstacle information $O_{BST}$ (7x7 grid with obstacle radius as values) as input and outputs N steps of actions $a_t$ (future position and future velocity).

<img src="assets/model_input.jpg" alt="drawing" width="480"/>

*The quadrotor icon is from [flaticon](https://www.flaticon.com/free-icon/quadcopter_5447794).


### Deviation from the original implementation
- Add a linear layer before the Mish activation to the condition encoder of `ConditionalResidualBlock1D`. This is to prevent the activation from truncating large negative values from the normalized observation.


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
- Even though the loss curve appears saturated, the performance of the controller can still improve as training continues.
  - The training loss curves of the diffusion model are extremely smooth btw.
- DDPM and DDIM samplers yield the best.
- Inference is not in real-time. The controller is set to sun 100Hz.

### Things that didn't work
- Tried encoding distances to each obstacle. Did not observe improvement in terms of collision avoidance.

