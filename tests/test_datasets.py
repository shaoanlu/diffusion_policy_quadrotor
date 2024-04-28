import torch
import unittest
import yaml

from core.dataset.quadrotor_dataset import PlanarQuadrotorStateDataset


class TestPlanarQuadrotorStateDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "tests/test_dataset.joblib"

        with open("tests/config/test_config.yaml", "r") as file:
            self.config = yaml.safe_load(file)

        self.obs_horizon = self.config["obs_horizon"]
        self.action_horizon = self.config["action_horizon"]
        self.pred_horizon = self.config["pred_horizon"]
        self.action_dim = self.config["controller"]["networks"]["action_dim"]
        self.obs_dim = self.config["controller"]["networks"]["obs_dim"]
        self.obstacle_encode_dim = self.config["controller"]["networks"]["obstacle_encode_dim"]

    def test_init(self):
        dataset = PlanarQuadrotorStateDataset(dataset_path=self.dataset_path, config=self.config)
        self.assertTrue(isinstance(dataset, PlanarQuadrotorStateDataset))

    def test_iter(self):
        batch_size = self.config["dataloader"]["batch_size"]
        dataset = PlanarQuadrotorStateDataset(dataset_path=self.dataset_path, config=self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        # batch context matches expectecd shapes
        batch = next(iter(dataloader))
        self.assertEqual(
            batch["obs"].shape, (batch_size, self.obs_dim * self.obs_horizon + self.obstacle_encode_dim)
        )
        self.assertEqual(batch["action"].shape, (batch_size, self.pred_horizon, self.action_dim))


if __name__ == "__main__":
    unittest.main()
