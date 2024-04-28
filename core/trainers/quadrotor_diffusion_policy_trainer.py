from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Dict, Optional

from core.controllers.quadrotor_diffusion_policy import build_noise_scheduler_from_config
from core.dataset.quadrotor_dataset import PlanarQuadrotorStateDataset
from core.trainers.base_trainer import BaseDiffusionPolicyTrainer
from utils.utils import get_device


def build_dataloader_from_dataset_and_config(config: Dict, dataset: torch.utils.data.Dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )


class PlanarQuadrotorDiffusionPolicyTrainer(BaseDiffusionPolicyTrainer):
    def __init__(
        self, net: nn.Module, dataset: PlanarQuadrotorStateDataset, config: Dict, device: Optional[str] = None
    ):
        self.net = net
        self.noise_scheduler = build_noise_scheduler_from_config(config)
        self.dataset = dataset
        self.set_config(config)
        self.device = get_device() if device is None else torch.device(device)

        self.net.to(self.device)

        # build optimizer
        if config["trainer"]["optimizer"]["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.net.parameters(),
                lr=config["trainer"]["optimizer"]["learning_rate"],
                weight_decay=config["trainer"]["optimizer"]["weight_decay"],
            )
        else:
            raise NotImplementedError

        # build dataset
        self.dataloader = build_dataloader_from_dataset_and_config(config, dataset)

        # set EMA
        self.use_ema = config["trainer"]["use_ema"]
        self.ema = EMAModel(parameters=self.net.parameters(), power=0.75) if self.use_ema else None

    def prepare_inputs(self, batch):
        # data normalized in dataset
        # device transfer
        obs_cond = batch["obs"].to(self.device, dtype=torch.float32)  # FiLM conditioning
        action = batch["action"].to(self.device, dtype=torch.float32)
        batch_size = obs_cond.shape[0]
        return obs_cond, action, batch_size

    def optimization_step(self, action, obs_cond, batch_size):
        # sample noise to add to actions
        noise = torch.randn(action.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        # predict the noise residual
        noise_pred = self.net(noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        if self.use_ema:
            self.ema.step(self.net.parameters())

        return loss

    def train(self, num_epochs: int, save_ckpt_epoch: int = None):
        if save_ckpt_epoch is None:
            save_ckpt_epoch = num_epochs

        # set learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name=self.config["trainer"]["lr_scheduler"]["name"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["trainer"]["lr_scheduler"]["num_warmup_steps"],
            num_training_steps=len(self.dataloader) * num_epochs,
        )

        # training loop
        trn_loss = []
        with tqdm(range(num_epochs), desc="Epoch") as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc="Batch", leave=False) as tepoch:
                    for nbatch in tepoch:
                        obs_cond, action, B = self.prepare_inputs(nbatch)
                        loss = self.optimization_step(action, obs_cond, B)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                trn_loss.append(np.mean(epoch_loss))

                # save intermediate ckpt
                if (epoch_idx + 1) % save_ckpt_epoch == 0:
                    self.save_checkpoint(path=f"ckpt_ep{epoch_idx}.ckpt")

        return trn_loss

    def save_checkpoint(self, path: str):
        save_model = self.net
        if self.config["trainer"]["use_ema"]:
            self.ema.copy_to(save_model.parameters())
        torch.save(save_model.state_dict(), path)

    def set_config(self, config: Dict):
        self.config = config
        self.obs_horizon = config["obs_horizon"]
        self.obs_dim = config["controller"]["networks"]["obs_dim"]
        self.action_horizon = config["action_horizon"]
        self.action_dim = config["controller"]["networks"]["action_dim"]
        self.pred_horizon = config["pred_horizon"]
        self.obstacle_encode_dim = config["controller"]["networks"]["obstacle_encode_dim"]
