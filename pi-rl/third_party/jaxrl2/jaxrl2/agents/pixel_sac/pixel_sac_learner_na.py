# adapted from jaxrl2
"""DSRL-NA (Noise-Aliased) Learner."""
import time
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent, get_batch_stats
from jaxrl2.agents.common import eval_actions_jit, sample_actions_jit, sample_actions_with_log_probs_jit
from jaxrl2.agents.pixel_sac.pixel_sac_learner import TrainState
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.agents.pixel_sac.noise_critic_updater import update_noise_critic, update_action_critic
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


@functools.partial(jax.jit, static_argnames=('color_jitter', 'aug_next', 'num_cameras'))
def _augment_batch_jit(
    rng: PRNGKey, batch: DatasetDict, color_jitter: bool, aug_next: bool, num_cameras: int,
) -> Tuple[PRNGKey, DatasetDict]:
    """Apply augmentation (crop + color jitter) to batch once."""
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    return rng, batch

@functools.partial(jax.jit)
def _update_temp_jit(
    rng: PRNGKey, actor: TrainState, temp: TrainState,
    batch: DatasetDict, target_entropy: float,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    key, rng = jax.random.split(rng)
    input_collections = {'params': actor.params}
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        input_collections['batch_stats'] = actor.batch_stats
    dist = actor.apply_fn(input_collections, batch['observations'])
    _, log_probs = dist.sample_and_log_prob(seed=key)
    entropy = -log_probs.mean()
    new_temp, alpha_info = update_temperature(temp, entropy, target_entropy)
    alpha_info['temperature_entropy'] = entropy
    return rng, new_temp, alpha_info

@functools.partial(jax.jit, static_argnames=('backup_entropy', 'critic_reduction', ))
def _update_action_critic_jit(
    rng: PRNGKey, action_critic: TrainState, target_action_critic_params: Params, 
    temp: TrainState,
    batch: DatasetDict, discount: float, tau: float,
    backup_entropy: bool,
    critic_reduction: str,) -> Tuple[PRNGKey, TrainState, Params, Dict[str, float]]:
    key, rng = jax.random.split(rng)
    target_action_critic = action_critic.replace(params=target_action_critic_params)
    new_action_critic, action_critic_info = update_action_critic(
        key,
        action_critic,
        target_action_critic,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
    )
    new_target_action_critic_params = soft_target_update(new_action_critic.params, target_action_critic_params, tau)

    return rng, new_action_critic, new_target_action_critic_params, action_critic_info

@functools.partial(jax.jit, static_argnames=('critic_reduction'))
def _update_noise_actor_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState, 
    temp: TrainState,
    batch: DatasetDict, discount: float, tau: float, target_entropy: float,
    # backup_entropy: bool,
    critic_reduction: str, noise_scale: float = 1.0,) -> Tuple[PRNGKey, TrainState, TrainState, Dict[str, float]]:
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, critic, temp, batch, critic_reduction=critic_reduction, noise_scale=noise_scale)
    return rng, new_actor, actor_info

@functools.partial(jax.jit, static_argnames=('critic_reduction'))
def _update_noise_critic_jit(
    rng: PRNGKey, critic: TrainState,
    action_critic: TrainState,
    batch: DatasetDict,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """Update noise critic and actor. Expects already-augmented batch.
    """
    key, rng = jax.random.split(rng)
    new_critic, critic_info = update_noise_critic(key, critic, action_critic, batch, critic_reduction=critic_reduction)
    return rng, new_critic, critic_info


class PixelSACLearnerNA(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 env_actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 backup_entropy: bool = False,
                 noise_scale_inside: bool = False,
                 ):
        """DSRL-NA agent with action-space and noise-space critics."""
        self.noise_scale_inside = noise_scale_inside
        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, action_critic_key, temp_key = jax.random.split(rng, 5)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        if self.noise_scale_inside:
            # Internal scaling: internal scaling to (-action_magnitude, action_magnitude)
            self.noise_scale = jnp.float32(1.0)
            policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim, dropout_rate=dropout_rate, low=-action_magnitude, high=action_magnitude)
        else:
            # External scaling
            self.noise_scale = jnp.float32(action_magnitude)
            policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim, dropout_rate=dropout_rate)

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)

        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(encoder=encoder_def,
                                      network=critic_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck
                                      )
        print(critic_def)
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        # Noise critic only does distillation, no TD learning, so no target network needed
        # target_critic_params = copy.deepcopy(critic_params)
        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)


        self._rng = rng
        self._actor = actor
        self._critic = critic
        # self._target_critic_params = target_critic_params
        self._temp = temp
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim
        else:
            self.target_entropy = float(target_entropy)
        self.backup_entropy = bool(backup_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(f'backup_entropy: {self.backup_entropy}')
        print(f'critic_reduction: {self.critic_reduction}')
        print(f"{self.tau=}, {self.discount=}, {self.aug_next=}, {self.color_jitter=}")
        time.sleep(5)

        action_critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        action_critic_def = PixelMultiplexer(encoder=encoder_def,
                                             network=action_critic_def,
                                             latent_dim=latent_dim,
                                             use_bottleneck=use_bottleneck
                                             )
        print(action_critic_def)
        action_critic_def_init = action_critic_def.init(action_critic_key, observations, env_actions)
        self._action_critic_init_params = action_critic_def_init['params']

        action_critic_params = action_critic_def_init['params']
        action_critic_batch_stats = action_critic_def_init['batch_stats'] if 'batch_stats' in action_critic_def_init else None
        
        self._action_critic = TrainState.create(apply_fn=action_critic_def.apply,
                                                params=action_critic_params,
                                                tx=optax.adam(learning_rate=critic_lr),
                                                batch_stats=action_critic_batch_stats
                                                )
        self._target_action_critic_params = copy.deepcopy(action_critic_params)
        
        print(f'[DSRL-NA] Noise action dim: {self.action_dim}, Env action Shape: {env_actions.shape}')
        print(f'[DSRL-NA] noise_scale_inside={self.noise_scale_inside}, noise_scale={self.noise_scale}, action_magnitude={action_magnitude}')

    def update(self, batch: FrozenDict, distill_batch: Optional[FrozenDict] = None, train_action_critic=True, train_noise_actor=True) -> Dict[str, float]:
        """Update all networks.
        
        Args:
            batch: Batch of data with observations, actions, rewards, etc.
            distill_batch: Optional batch with distillation data (distill_noise, distill_actions)
        Returns:
            Info dict with training metrics.
        """
        info = {}
        
        # Augment batch once, use for both critics
        self._rng, aug_batch = _augment_batch_jit(
            self._rng, batch, self.color_jitter, self.aug_next, self.num_cameras)

        # Update action-space critic       
        if train_action_critic:
            self._rng, self._temp, temp_info = _update_temp_jit(
                self._rng, self._actor, self._temp, aug_batch, self.target_entropy
            )
            info.update(temp_info)
            self._rng, self._action_critic, self._target_action_critic_params, action_critic_info = _update_action_critic_jit(
                self._rng, self._action_critic, self._target_action_critic_params, self._temp, aug_batch, self.discount, self.tau,
                self.backup_entropy, self.critic_reduction,
            )
            info.update(action_critic_info)
        
        # Update noise-space actor        
        if train_noise_actor:
            self._rng, self._actor, noise_actor_info = _update_noise_actor_jit(
                self._rng, self._actor, self._critic, self._temp, aug_batch, self.discount, self.tau, self.target_entropy, self.critic_reduction, self.noise_scale
                )
            info.update(noise_actor_info)

        # Add distillation data if provided
        if distill_batch is not None:
            aug_batch = aug_batch.copy(add_or_replace={'distill_noise': distill_batch['distill_noise'], 'distill_actions': distill_batch['distill_actions']})
            self._rng, self._critic, noise_critic_info = _update_noise_critic_jit(
                self._rng, self._critic, self._action_critic, aug_batch, critic_reduction=self.critic_reduction
            )
            info.update(noise_critic_info)
        return info

    def perform_eval(self, variant, i, eval_buffer, eval_buffer_iterator, eval_env):
        """Return value/reward visualization images for `eval_buffer`.

        Caller is responsible for logging (e.g. wrapping with wandb.Image). Keeps
        jaxrl2 free of wandb and cross-package imports back into examples/.
        """
        trajs = eval_buffer.get_random_trajs(3)
        return self.make_value_reward_visulization(variant, trajs)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        tqdm.write('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'actor': self._actor,
            'temp': self._temp,
            'action_critic': self._action_critic,
            'target_action_critic_params': self._target_action_critic_params,
        }
        return save_dict

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self._actor.apply_fn, self._actor.params,
                                   observations, get_batch_stats(self._actor))
        return np.asarray(actions) * float(self.noise_scale)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(self._rng, self._actor.apply_fn,
                                          self._actor.params, observations, get_batch_stats(self._actor))
        self._rng = rng
        actions = np.clip(actions, -1, 1)
        return np.asarray(actions) * float(self.noise_scale)

    def sample_actions_with_log_probs(self, observations: np.ndarray):
        """Sample actions, compute log_prob on unscaled noise, then scale."""
        rng, actions, log_probs = sample_actions_with_log_probs_jit(
            self._rng, self._actor.apply_fn,
            self._actor.params, observations, get_batch_stats(self._actor))
        self._rng = rng
        actions = np.clip(actions, -1, 1)
        return np.asarray(actions) * float(self.noise_scale), np.asarray(log_probs)

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._temp = output_dict['temp']
        if 'action_critic' in output_dict:
            self._action_critic = output_dict['action_critic']
            self._target_action_critic_params = output_dict['target_action_critic_params']
        print('restored from ', dir)
        
    
@functools.partial(jax.jit)
def get_value(action, observation, critic):
    input_collections = {'params': critic.params}
    q_pred = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, masks, images):

    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., :3, -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image