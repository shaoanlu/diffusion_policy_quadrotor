# diffusion_policy_quadrotor
This repository provides a demonstration of imitation learning using a diffusion policy. The implementation is adapted from the official Diffusion Policy [repository](https://github.com/real-stanford/diffusion_policy).

## Result
The animation shows the denoising process of the diffusion policy predicting future trajectory and applying actions.

<img src="assets/result_anim.gif" alt="drawing" width="380"/> <img src="assets/result_plot.png" alt="drawing" width="370"/>


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
The policy takes 1) the latest N step of observation (position and velocity) and 2) encoding of obstacle information (7x7 grid with obstacle radius as values) as input and outputs N steps of actions (future position and future velocity).

<img src="assets/model_input.jpg" alt="drawing" width="480"/>
