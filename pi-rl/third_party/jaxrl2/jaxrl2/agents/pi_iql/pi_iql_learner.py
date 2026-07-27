# adapted from jaxrl2
"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

from jaxrl2.agents.agent import Agent
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pi_iql.critic_updater import update_q, update_v
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.values import StateActionEnsemble, StateValue
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def shard_batch(batch):
    n_devices = jax.local_device_count()

    def _shard(x):
        x = jnp.asarray(x)
        assert x.shape[0] % n_devices == 0, (
            f"Batch size {x.shape[0]} must be divisible by num devices {n_devices}"
        )
        return x.reshape((n_devices, x.shape[0] // n_devices) + x.shape[1:])

    return jax.tree_map(_shard, batch)


def _apply_critic(critic: TrainState, observations, actions):
    if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
        return critic.apply_fn(
            {"params": critic.params, "batch_stats": critic.batch_stats},
            observations,
            actions,
            mutable=False,
        )
    return critic.apply_fn(
        {"params": critic.params},
        observations,
        actions,
    )


def _apply_value(value: TrainState, observations):
    if hasattr(value, "batch_stats") and value.batch_stats is not None:
        return value.apply_fn(
            {"params": value.params, "batch_stats": value.batch_stats},
            observations,
            mutable=False,
        )
    return value.apply_fn(
        {"params": value.params},
        observations,
    )


def _compute_adv_weights(
    critic: TrainState,
    value: TrainState,
    batch: FrozenDict,
    A_scaling: float,
    critic_reduction: str,
):
    qs = _apply_critic(critic, batch["observations"], batch["actions"])

    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise ValueError(f"Invalid critic reduction: {critic_reduction}")

    v = _apply_value(value, batch["observations"])

    adv = q - v
    raw_weight = jnp.exp(A_scaling * adv)
    advs = jnp.minimum(raw_weight, 100.0)

    info = {
        "adv_mean": adv.mean(),
        "adv_std": adv.std(),
        "adv_min": adv.min(),
        "adv_max": adv.max(),
        "adv_pos_frac": (adv > 0).astype(jnp.float32).mean(),
        "adv_weight_mean": advs.mean(),
        "adv_weight_std": advs.std(),
        "adv_weight_max": advs.max(),
        # fraction of samples whose unclipped exp(A * adv) hit the 100 ceiling
        "adv_clip_frac": (raw_weight >= 100.0).astype(jnp.float32).mean(),
        "q_in_advs": q.mean(),
        "q_in_advs_std": q.std(),
        "v_in_advs": v.mean(),
        "v_in_advs_std": v.std(),
    }
    # Per-sample tensors returned OUTSIDE info so they survive pmean reductions.
    # qs: [num_qs, B_local], q: [B_local], v: [B_local], adv: [B_local], advs: [B_local]
    return advs, info, qs, v, adv


def _update_step(
    rng: PRNGKey,
    critic: TrainState,
    target_critic_params: Params,
    value: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    expectile: float,
    A_scaling: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
) -> Tuple[PRNGKey, TrainState, Params, TrainState, Dict[str, float]]:
    aug_pixels = batch["observations"]["pixels"]
    aug_next_pixels = batch["next_observations"]["pixels"]

    if batch["observations"]["pixels"].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch["observations"]["pixels"])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:, :, :, i * 3:(i + 1) * 3].set(
                        (
                            color_transform(
                                key,
                                aug_pixels[:, :, :, i * 3:(i + 1) * 3].astype(jnp.float32) / 255.0,
                            )
                            * 255
                        ).astype(jnp.uint8)
                    )
            else:
                aug_pixels = (
                    color_transform(key, aug_pixels.astype(jnp.float32) / 255.0) * 255
                ).astype(jnp.uint8)

    observations = batch["observations"].copy(add_or_replace={"pixels": aug_pixels})
    batch = batch.copy(add_or_replace={"observations": observations})

    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch["next_observations"]["pixels"])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:, :, :, i * 3:(i + 1) * 3].set(
                        (
                            color_transform(
                                key,
                                aug_next_pixels[:, :, :, i * 3:(i + 1) * 3].astype(jnp.float32) / 255.0,
                            )
                            * 255
                        ).astype(jnp.uint8)
                    )
            else:
                aug_next_pixels = (
                    color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.0) * 255
                ).astype(jnp.uint8)

        next_observations = batch["next_observations"].copy(
            add_or_replace={"pixels": aug_next_pixels}
        )
        batch = batch.copy(add_or_replace={"next_observations": next_observations})

    target_critic = critic.replace(params=target_critic_params)

    new_value, value_info = update_v(
        target_critic,
        value,
        batch,
        expectile,
        critic_reduction,
        axis_name="data",
    )

    new_critic, critic_info = update_q(
        critic,
        new_value,
        batch,
        discount,
        axis_name="data",
    )

    new_target_critic_params = soft_target_update(
        new_critic.params,
        target_critic_params,
        tau,
    )

    info = {**critic_info, **value_info}
    info = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="data"), info)

    return rng, new_critic, new_target_critic_params, new_value, info


def _compute_advs_step(
    critic: TrainState,
    value: TrainState,
    batch: FrozenDict,
    A_scaling: float,
    critic_reduction: str,
):
    advs, info, qs, v, adv = _compute_adv_weights(
        critic,
        value,
        batch,
        A_scaling,
        critic_reduction,
    )
    info = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="data"), info)
    return advs, info, qs, v, adv


class PiIQLLearner(Agent):

    def __init__(
        self,
        seed: int,
        observations: Union[jnp.ndarray, DatasetDict],
        actions: jnp.ndarray,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        A_scaling: float = 1.0,
        critic_reduction: str = "mean",
        encoder_type="resnet_34_v1",
        encoder_norm="group",
        color_jitter=True,
        use_spatial_softmax=True,
        softmax_temperature=1,
        aug_next=True,
        use_bottleneck=True,
        num_qs: int = 2,
        num_cameras: int = 1,
        action_horizon: int = 1,
    ):
        self.aug_next = aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        # Bellman target spans an H-step skip (s_t -> s_{t+H}), so the
        # next-state discount in update_q is gamma^H, while self.discount stays
        # the per-step gamma used to build R = Σ γ^h r_h on the data side.
        self.action_horizon = action_horizon
        self.bellman_discount = float(discount ** action_horizon)
        self.expectile = expectile
        self.A_scaling = A_scaling
        self.critic_reduction = critic_reduction

        self.num_devices = jax.local_device_count()
        print(f"Using {self.num_devices} local devices for PixelIQLLearner DDP.")

        rng = jax.random.PRNGKey(seed)
        rng, critic_key, value_key = jax.random.split(rng, 3)

        if encoder_type == "small":
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == "impala":
            print("using impala")
            encoder_def = ImpalaEncoder()
        elif encoder_type == "impala_small":
            print("using impala small")
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == "resnet_small":
            encoder_def = ResNetSmall(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_18_v1":
            encoder_def = ResNet18(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_34_v1":
            encoder_def = ResNet34(
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
            )
        elif encoder_type == "resnet_small_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == "resnet_18_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == "resnet_34_v2":
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError("encoder type not found!")

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])

        critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        print(critic_def)

        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init["params"]

        critic_params = critic_def_init["params"]
        critic_batch_stats = critic_def_init["batch_stats"] if "batch_stats" in critic_def_init else None
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=critic_batch_stats,
        )
        target_critic_params = copy.deepcopy(critic_params)

        value_def = StateValue(hidden_dims)
        value_def = PixelMultiplexer(
            encoder=encoder_def,
            network=value_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
        )
        print(value_def)

        value_def_init = value_def.init(value_key, observations)
        value_params = value_def_init["params"]
        value_batch_stats = value_def_init["batch_stats"] if "batch_stats" in value_def_init else None
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=value_lr),
            batch_stats=value_batch_stats,
        )

        self._rng = jax.random.split(rng, self.num_devices)
        self._critic = replicate(critic)
        self._target_critic_params = replicate(target_critic_params)
        self._value = replicate(value)

        self._update_pmap = jax.pmap(
            functools.partial(
                _update_step,
                discount=self.bellman_discount,
                tau=self.tau,
                expectile=self.expectile,
                A_scaling=self.A_scaling,
                critic_reduction=self.critic_reduction,
                color_jitter=self.color_jitter,
                aug_next=self.aug_next,
                num_cameras=self.num_cameras,
            ),
            axis_name="data",
        )

        self._compute_advs_pmap = jax.pmap(
            functools.partial(
                _compute_advs_step,
                A_scaling=self.A_scaling,
                critic_reduction=self.critic_reduction,
            ),
            axis_name="data",
        )

        print(self.critic_reduction)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        batch = shard_batch(batch)

        new_rng, new_critic, new_target_critic, new_value, info = self._update_pmap(
            self._rng,
            self._critic,
            self._target_critic_params,
            self._value,
            batch,
        )

        self._rng = new_rng
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        self._value = new_value

        info = jax.tree_map(lambda x: x[0], info)
        return info

    def compute_advs(self, batch: FrozenDict, output_sharding=None):
        """Compute IQL advantage weights for the global batch.

        Args:
            batch: Unsharded global batch.
            output_sharding: Optional ``jax.sharding.Sharding`` to place the
                device-side advs onto. When provided, the returned
                ``advs_device`` is already in that sharding so callers can feed
                it straight into a jit-compiled step without an extra
                host→device transfer. When ``None``, the pmap-style sharded
                array is returned as-is.

        Returns:
            advs_device: jax.Array with shape [global_batch], live on device.
            advs_host: numpy array with shape [global_batch] (for host stats).
            info: aggregated scalar logging info.
            qs: per-sample ensemble Q values, shape [num_qs, global_batch].
            v: per-sample value, shape [global_batch].
            adv: per-sample (q - v), shape [global_batch].
        """
        batch = shard_batch(batch)
        advs, info, qs, v, adv = self._compute_advs_pmap(
            self._critic,
            self._value,
            batch,
        )

        # Gather [n_dev, B/n_dev, ...] -> [B, ...] on device (no host roundtrip).
        advs_device = advs.reshape((-1,) + advs.shape[2:])
        if output_sharding is not None:
            advs_device = jax.device_put(advs_device, output_sharding)

        # Host copies (needed for numpy stats / wandb histograms).
        advs_host = np.asarray(jax.device_get(advs_device))
        # qs shape after pmap: [n_dev, num_qs, B/n_dev] -> [num_qs, B]
        qs_h = jax.device_get(qs)
        qs_h = jnp.transpose(qs_h, (1, 0, 2)).reshape((qs_h.shape[1], -1))
        v_h = jax.device_get(v).reshape(-1)
        adv_h = jax.device_get(adv).reshape(-1)

        info = jax.tree_map(lambda x: x[0], info)
        return advs_device, advs_host, info, np.asarray(qs_h), np.asarray(v_h), np.asarray(adv_h)

    def perform_eval(self, variant, i, eval_buffer, eval_buffer_iterator, eval_env):
        """Return value/reward visualization images for `eval_buffer`.

        Caller is responsible for logging (e.g. wrapping with wandb.Image). Keeps
        jaxrl2 free of wandb and cross-package imports back into examples/.
        """
        trajs = eval_buffer.get_random_trajs(3)
        return self.make_value_reward_visulization(variant, trajs)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs["rewards"])
        traj_images = []

        critic = unreplicate(self._critic)

        for itraj in range(num_traj):
            observations = trajs["observations"][itraj]
            next_observations = trajs["next_observations"][itraj]
            actions = trajs["actions"][itraj]
            rewards = trajs["rewards"][itraj]
            masks = trajs["masks"][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations["pixels"][t]
                next_obs_pixels = next_observations["pixels"][t]

                obs_dict = {"pixels": obs_pixels[None]}
                for k, v in observations.items():
                    if "pixels" not in k:
                        obs_dict[k] = v[t][None]

                next_obs_dict = {"pixels": next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if "pixels" not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations["pixels"]))

        print("finished reward value visuals.")
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            "critic": unreplicate(self._critic),
            "target_critic_params": unreplicate(self._target_critic_params),
            "value": unreplicate(self._value),
        }
        return save_dict

    def iql_save_pytree(self):
        """Host-side pytree of critic / target_critic_params / value for orbax.

        Mirrors ``_save_dict`` but exposed as a regular method so trainers can
        hand it to an external CheckpointManager (e.g. scripts/train_iql.py
        wires this through openpi.training.checkpoints).
        """
        return {
            "critic": unreplicate(self._critic),
            "target_critic_params": unreplicate(self._target_critic_params),
            "value": unreplicate(self._value),
        }

    def iql_load_pytree(self, restored: dict) -> None:
        """Replace critic / target / value state from a restored pytree.

        Re-replicates each leaf across local devices so the pmap update paths
        keep working without further setup.
        """
        self._critic = replicate(restored["critic"])
        self._target_critic_params = replicate(restored["target_critic_params"])
        self._value = replicate(restored["value"])

    def save_checkpoint(self, dir, step):
        checkpoints.save_checkpoint(
            ckpt_dir=dir,
            target=self._save_dict,
            step=step,
            overwrite=True,
            keep=3,
        )

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)

        self._critic = replicate(output_dict["critic"])
        self._target_critic_params = replicate(output_dict["target_critic_params"])
        self._value = replicate(output_dict["value"])

        print("restored from ", dir)


@functools.partial(jax.jit)
def get_value(action, observation, critic):
    if hasattr(critic, "batch_stats") and critic.batch_stats is not None:
        input_collections = {"params": critic.params, "batch_stats": critic.batch_stats}
    else:
        input_collections = {"params": critic.params}
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
    images = images[..., -1]
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle="--", marker="o")
    else:
        axs[1].plot(q_estimates_np, linestyle="--", marker="o")
    axs[1].set_ylabel("q values")

    axs[2].plot(rewards, linestyle="--", marker="o")
    axs[2].set_ylabel("rewards")
    axs[2].set_xlim([0, len(rewards)])

    axs[3].plot(masks, linestyle="--", marker="d")
    axs[3].set_ylabel("masks")
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image
