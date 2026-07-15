from __future__ import annotations

from typing import Any, Optional

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import SACPolicy


class SACActor(BasePolicy):
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        del net_arch, activation_fn, log_std_init, full_std, use_expln, clip_mean
        if use_sde:
            raise NotImplementedError("SACActor does not support gSDE.")

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        action_dim = get_action_dim(self.action_space)
        if features_dim != 11:
            raise ValueError(f"SACActor expects obs dim 11, got {features_dim}.")
        if action_dim != 1:
            raise ValueError(f"SACActor expects action dim 1, got {action_dim}.")

        self.features_dim = features_dim
        self.trunk = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.std = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.ReLU(),
            nn.Softplus(),
        )
        self.std_scale = 1.0
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=[],
                features_dim=self.features_dim,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.trunk(features)
        mean_actions = self.mu(latent_pi)
        std_actions = self.std(latent_pi) * self.std_scale
        log_std = th.log(std_actions)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_path = nn.Linear(obs_dim, 128)
        self.action_path = nn.Linear(action_dim, 128)
        self.post_concat = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs_features: th.Tensor, actions: th.Tensor) -> th.Tensor:
        obs_latent = self.obs_path(obs_features)
        action_latent = self.action_path(actions)
        return self.post_concat(th.cat([obs_latent, action_latent], dim=1))


class SACCritic(BaseModel):
    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        del net_arch, activation_fn
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        if features_dim != 11:
            raise ValueError(f"SACCritic expects obs dim 11, got {features_dim}.")
        if action_dim != 1:
            raise ValueError(f"SACCritic expects action dim 1, got {action_dim}.")
        if n_critics != 2:
            raise ValueError(f"SACCritic expects exactly 2 critics, got {n_critics}.")

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = QNetwork(features_dim, action_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


class MatlabSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[list[int] | dict[str, list[int]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[] if net_arch is None else net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SACActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SACActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SACCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return SACCritic(**critic_kwargs).to(self.device)
