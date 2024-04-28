import torch
import unittest
import yaml

from core.networks.conditional_unet1d import ConditionalUnet1D


class TestConditionalUnet1D(unittest.TestCase):
    def setUp(self):
        with open("tests/config/test_config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

        self.obs_horizon = self.config["obs_horizon"]
        self.action_horizon = self.config["action_horizon"]
        self.pred_horizon = self.config["pred_horizon"]
        self.action_dim = self.config["controller"]["networks"]["action_dim"]
        self.obs_dim = self.config["controller"]["networks"]["obs_dim"]
        self.obstacle_encode_dim = self.config["controller"]["networks"]["obstacle_encode_dim"]

    def test_init(self):
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.obs_dim * self.obs_horizon + self.obstacle_encode_dim
        )
        self.assertTrue(isinstance(noise_pred_net, ConditionalUnet1D))

    def test_inference(self):
        net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.obs_dim * self.obs_horizon + self.obstacle_encode_dim
        )

        # example inputs
        noised_action = torch.randn((1, self.pred_horizon, self.action_dim))
        obs = torch.zeros((1, self.obs_horizon * self.obs_dim + self.obstacle_encode_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = net(sample=noised_action, timestep=diffusion_iter, global_cond=obs.flatten(start_dim=1))
        # removing noise
        denoised_action = noised_action - noise

        self.assertEqual(denoised_action.shape, (self.action_horizon, self.action_dim))


if __name__ == "__main__":
    unittest.main()
