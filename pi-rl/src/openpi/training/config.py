# adapted from openpi
"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.franka_single_policy as franka_single_policy
import openpi.policies.franka_dual_policy as franka_dual_policy
import openpi.policies.franka_optimized_policy as franka_optimized_policy
import openpi.policies.robotwin_policy as robotwin_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.episode_filter as _episode_filter
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional episode whitelist. For single-repo datasets: tuple of episode indices.
    # For multi-repo (MultiDataConfig): mapping of {repo_id: tuple[int, ...]}.
    episode_ids: tuple[int, ...] | Mapping[str, tuple[int, ...]] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


@dataclasses.dataclass(frozen=True)
class MultiDataConfig(DataConfig):
    """DataConfig variant describing a multi-task dataset (multiple LeRobot repos joined).

    The inherited ``repo_id`` field is used purely as an asset / norm-stats identifier here --
    the actual data load is driven by ``repo_ids``. Consumers distinguish single- vs multi-task
    loaders by ``isinstance(cfg, MultiDataConfig)``.
    """

    # LeRobot repo ids to combine via MultiLeRobotDataset.
    repo_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RLDataConfig(DataConfig):
    """DataConfig variant for offline RL training.

    Carries the same transform chain as a standard pi0 DataConfig, but signals
    via its type that each sample is a transition ``(s, a, s', r, mask)`` rather
    than a supervised ``(s, a)`` pair. Consumers (e.g. ``rl_data_loader``)
    distinguish RL vs supervised data by ``isinstance(cfg, RLDataConfig)``.

    The transition shape itself is encoded in ``extra_delta_timestamps`` (the
    factory sets up state/image/wrist_image at ``[0, H/fps]`` and progress at
    ``[0, 1/fps, ..., H/fps]`` so each lerobot lookup carries both current and
    next-step views). ``reward_source`` records how the per-sample reward was
    sourced and lets ``create_rl_dataset`` sanity-check the pipeline.
    """

    # Extra ``delta_timestamps`` entries to merge into the LeRobot dataset call (in addition to the
    # action chunk implied by ``action_sequence_keys``). Used to ask lerobot for next-step
    # observations alongside the current frame — e.g. IQL passes
    # ``state``/``image``/``wrist_image``/``progress`` here as ``(0, H/fps)`` so each sample
    # carries both the current and next-step views needed to form a transition.
    extra_delta_timestamps: dict[str, tuple[float, ...]] = dataclasses.field(default_factory=dict)

    # Where the per-sample reward comes from:
    #   'progress': read from the dataset's ``progress`` feature (dense reward = endpoint
    #               delta over the action chunk). Requires ``progress`` in lerobot meta
    #               features and a matching ``extra_delta_timestamps['progress']`` entry.
    #   'terminal': sparse r=1 when the action chunk would run past the episode end,
    #               0 otherwise.
    reward_source: Literal["progress", "terminal"] = "progress"


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class MultiDataConfigFactory(DataConfigFactory):
    """Base factory for multi-task dataset configs.

    Subclasses populate ``repo_ids`` either directly or by resolving some convenience field
    (e.g. RoboTwin's ``robotwin_root`` glob) inside ``create()``. The factory's ``repo_id``
    is used purely as the asset / norm-stats identifier -- it is *not* a real LeRobot path.
    """

    # Concrete dataset paths joined via MultiLeRobotDataset.
    repo_ids: tuple[str, ...] = ()

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        """Create a multi-task data config."""

    def create_base_multi_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig, repo_ids: Sequence[str]
    ) -> MultiDataConfig:
        """Like ``create_base_config`` but returns a MultiDataConfig carrying ``repo_ids``."""
        single = self.create_base_config(assets_dirs, model_config)
        # Promote DataConfig -> MultiDataConfig (all DataConfig fields carry over).
        return MultiDataConfig(
            **{f.name: getattr(single, f.name) for f in dataclasses.fields(single)},
            repo_ids=tuple(repo_ids),
        )


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoRLDataConfig(DataConfigFactory):
    """Libero data config for offline RL training (used by IQL and any future RL alg).

    Two differences vs. :class:`LeRobotLiberoDataConfig`:
      * Sets ``extra_delta_timestamps`` so lerobot returns state/image/wrist_image
        as a 2-frame stack ``[t, t+H]`` and (when ``reward_source == 'progress'``)
        ``progress`` as an ``H+1`` sequence ``[t, t+1, ..., t+H]`` per sample —
        one lerobot lookup gives the full transition.
      * Plugs in :class:`openpi.policies.libero_policy.LiberoRLInputs` instead
        of the standard ``LiberoInputs``: it splits the stacked observations,
        packs the current view in pi0 format, exposes next-step pixels/state as
        side-channel keys, and emits a per-sample reward + mask.

    Produces an :class:`RLDataConfig`, which the RL data loader type-checks for.
    """

    # How the reward is sourced; see :class:`RLDataConfig`. ``'auto'`` picks
    # ``'progress'`` when the lerobot dataset has a ``progress`` feature, else
    # falls back to ``'terminal'``.
    reward_source: Literal["auto", "progress", "terminal"] = "auto"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        # Imported lazily to keep module-load fast (lerobot pulls a lot in).
        import openpi.training.lerobot_patched as _lerobot_patched

        meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(self.repo_id)
        fps = float(meta.fps)
        horizon = model_config.action_horizon
        next_dt = horizon / fps
        feature_keys = set(meta.features.keys())
        has_progress = "progress" in feature_keys

        # Resolve the reward source up front so we can shape extra_delta_timestamps
        # and the repack structure consistently with what RLLiberoInputs will see.
        if self.reward_source == "auto":
            resolved_reward: Literal["progress", "terminal"] = "progress" if has_progress else "terminal"
        elif self.reward_source == "progress":
            if not has_progress:
                raise ValueError(
                    f"reward_source='progress' requires a 'progress' feature in {self.repo_id}, "
                    f"but found only: {sorted(feature_keys)}"
                )
            resolved_reward = "progress"
        else:
            resolved_reward = "terminal"

        extra_dt: dict[str, tuple[float, ...]] = {
            "state": (0.0, next_dt),
            "image": (0.0, next_dt),
            "wrist_image": (0.0, next_dt),
        }
        if resolved_reward == "progress":
            # H+1 samples covering every step of the action chunk plus its endpoint:
            # [progress(t), progress(t+1), ..., progress(t+H)]. Lets the trainer
            # look at the full per-step reward trajectory, not just the endpoint.
            extra_dt["progress"] = tuple(t / fps for t in range(horizon + 1))

        # Repack lerobot keys -> the dotted paths LiberoRLInputs reads. Only
        # include ``progress`` keys when we're actually sourcing reward from
        # progress, so RepackTransform won't KeyError on legacy datasets.
        structure: dict = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "observation/state_is_pad": "state_is_pad",
            "actions": "actions",
            "actions_is_pad": "actions_is_pad",
            "prompt": "prompt",
        }
        if resolved_reward == "progress":
            structure["progress"] = "progress"
            structure["progress_is_pad"] = "progress_is_pad"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(structure)],
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoRLInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        # Splat the base DataConfig into an RLDataConfig and override the
        # transform / delta_timestamps / reward fields.
        base = self.create_base_config(assets_dirs, model_config)
        overrides = {
            "extra_delta_timestamps": extra_dt,
            "repack_transforms": repack_transform,
            "data_transforms": data_transforms,
            "model_transforms": model_transforms,
            "reward_source": resolved_reward,
        }
        carried = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name not in overrides
        }
        return RLDataConfig(**carried, **overrides)


# Default RynnValue episode-filter path shared by every RoboTwin multi-task config below.
# Files that are absent at load time fall back to "use all episodes" with a warning --
# see episode_filter.load_or_resolve. So this default is safe to ship even when the
# central filter.json isn't mounted (e.g. local eval boxes).
ROBOTWIN_EPISODE_FILTER = "/path/to/robotwin_episode_filter/filter.json"


def _discover_robotwin_subtasks(root: str | pathlib.Path) -> tuple[str, ...]:
    """Return the absolute paths of every per-task RoboTwin sub-dir under ``root``.

    A sub-dir qualifies if it contains a ``meta/info.json`` (i.e. is a LeRobot dump).
    The list is sorted for determinism.
    """
    root_path = pathlib.Path(root)
    if not root_path.exists():
        return ()
    subs = []
    for sub in sorted(root_path.iterdir()):
        if sub.is_dir() and (sub / "meta" / "info.json").exists():
            subs.append(str(sub))
    return tuple(subs)


# Shared transform groups for the RoboTwin schema. Reused by the single- and multi-task
# factories so the data pipeline stays identical -- only the dataset source differs.
def _robotwin_transforms(
    model_config: _model.BaseModelConfig,
    *,
    use_delta_xyz_actions: bool,
    default_prompt: str | None,
) -> tuple[_transforms.Group, _transforms.Group, _transforms.Group]:
    repack_transform = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )
    data_transforms = _transforms.Group(
        inputs=[robotwin_policy.RobotwinInputs()],
        outputs=[robotwin_policy.RobotwinOutputs()],
    )
    if use_delta_xyz_actions:
        # Per arm: 3 xyz delta dims, then 5 absolute dims (4 quat + 1 gripper). Two arms.
        delta_action_mask = _transforms.make_bool_mask(3, -5, 3, -5)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )
    model_transforms = ModelTransformFactory(default_prompt=default_prompt)(model_config)
    return repack_transform, data_transforms, model_transforms


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinDataConfig(DataConfigFactory):
    """Single-task data config for one RoboTwin lerobot dump.

    Use this when fine-tuning on exactly one sub-task. For training jointly on multiple
    sub-tasks use :class:`LeRobotRobotwinMultiDataConfig` instead.
    """

    # Convert per-arm xyz translation dims of the action chunk into deltas relative to the
    # current state. Quaternions and gripper stay absolute.
    use_delta_xyz_actions: bool = False
    # Optional default prompt (overrides the per-episode task instruction when set).
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform, data_transforms, model_transforms = _robotwin_transforms(
            model_config,
            use_delta_xyz_actions=self.use_delta_xyz_actions,
            default_prompt=self.default_prompt,
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinRLDataConfig(DataConfigFactory):
    """RoboTwin data config for offline RL training (used by IQL and any future RL alg).

    Same shape as :class:`LeRobotRobotwinDataConfig` for the supervised pi0 view
    (cam_high → base_0_rgb, cam_left_wrist / cam_right_wrist → wrists, 16-dim
    state/action). Differences:
      * Sets ``extra_delta_timestamps`` so lerobot returns state and all three
        camera streams as a 2-frame stack ``[t, t+H]``, and (when
        ``reward_source == 'progress'``) progress as an ``H+1`` sequence
        ``[t, t+1, ..., t+H]`` — one lookup gives the full transition.
      * Plugs in :class:`openpi.policies.robotwin_policy.RobotwinRLInputs`
        instead of the standard ``RobotwinInputs``.

    Produces an :class:`RLDataConfig`, which the RL data loader type-checks for.
    """

    # See LeRobotRobotwinDataConfig for the action-space knob.
    use_delta_xyz_actions: bool = False
    # Optional default prompt (overrides the per-episode task instruction when set).
    default_prompt: str | None = None
    # See LeRobotLiberoRLDataConfig.reward_source.
    reward_source: Literal["auto", "progress", "terminal"] = "auto"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        # Imported lazily to keep module-load fast (lerobot pulls a lot in).
        import openpi.training.lerobot_patched as _lerobot_patched

        meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(self.repo_id)
        fps = float(meta.fps)
        horizon = model_config.action_horizon
        next_dt = horizon / fps
        feature_keys = set(meta.features.keys())
        has_progress = "progress" in feature_keys

        # Resolve reward source up front -- shapes extra_delta_timestamps and repack.
        if self.reward_source == "auto":
            resolved_reward: Literal["progress", "terminal"] = "progress" if has_progress else "terminal"
        elif self.reward_source == "progress":
            if not has_progress:
                raise ValueError(
                    f"reward_source='progress' requires a 'progress' feature in {self.repo_id}, "
                    f"but found only: {sorted(feature_keys)}"
                )
            resolved_reward = "progress"
        else:
            resolved_reward = "terminal"

        extra_dt: dict[str, tuple[float, ...]] = {
            "observation.state": (0.0, next_dt),
            "observation.images.cam_high": (0.0, next_dt),
            "observation.images.cam_left_wrist": (0.0, next_dt),
            "observation.images.cam_right_wrist": (0.0, next_dt),
        }
        if resolved_reward == "progress":
            extra_dt["progress"] = tuple(t / fps for t in range(horizon + 1))

        # Repack lerobot keys -> the dotted paths RobotwinRLInputs reads. Only
        # include progress keys when we're sourcing reward from progress.
        structure: dict = {
            "images": {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "actions_is_pad": "action_is_pad",
            "prompt": "prompt",
            "state_is_pad": "observation.state_is_pad",
        }
        if resolved_reward == "progress":
            structure["progress"] = "progress"
            structure["progress_is_pad"] = "progress_is_pad"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(structure)])
        data_transforms = _transforms.Group(
            inputs=[robotwin_policy.RobotwinRLInputs()],
            outputs=[robotwin_policy.RobotwinOutputs()],
        )
        if self.use_delta_xyz_actions:
            # Per arm: 3 xyz delta dims, then 5 absolute dims (4 quat + 1 gripper). Two arms.
            delta_action_mask = _transforms.make_bool_mask(3, -5, 3, -5)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        overrides = {
            "extra_delta_timestamps": extra_dt,
            "repack_transforms": repack_transform,
            "data_transforms": data_transforms,
            "model_transforms": model_transforms,
            "action_sequence_keys": ("action",),  # RoboTwin uses singular "action".
            "reward_source": resolved_reward,
        }
        carried = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name not in overrides
        }
        return RLDataConfig(**carried, **overrides)


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinMultiDataConfig(MultiDataConfigFactory):
    """Multi-task RoboTwin data config -- joins multiple lerobot dumps via MultiLeRobotDataset.

    Either set ``repo_ids`` directly or set ``robotwin_root`` to a directory containing one
    lerobot dump per task; the factory globs the root and uses every sub-dir with a
    ``meta/info.json``. ``repo_id`` is the asset / norm-stats identifier (e.g. ``robotwin_all``).

    Set ``episode_filter_path`` to a RynnValue ``filter.json`` to restrict training to a
    whitelist of episodes per sub-task. The factory prefers a precomputed compact cache at
    ``<assets_dirs>/<asset_id>/episodes.json`` (much faster than re-parsing the 24 MB filter
    every run); :mod:`compute_norm_stats_fast` writes that cache as a side effect.
    """

    use_delta_xyz_actions: bool = False
    default_prompt: str | None = None
    # Convenience: if ``repo_ids`` is empty, discover lerobot sub-dumps under this root.
    robotwin_root: str | None = None
    # Optional RynnValue filter.json restricting which episodes are used.
    episode_filter_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        repo_ids = self.repo_ids
        if not repo_ids:
            if not self.robotwin_root:
                raise ValueError("LeRobotRobotwinMultiDataConfig needs `repo_ids` or `robotwin_root`.")
            repo_ids = _discover_robotwin_subtasks(self.robotwin_root)
            if not repo_ids:
                raise ValueError(f"robotwin_root={self.robotwin_root!r} contains no lerobot sub-dumps.")
            logging.info(
                f"RoboTwin multi-task: discovered {len(repo_ids)} sub-tasks under {self.robotwin_root}"
            )

        # Resolve the episode whitelist (if any). Prefer the compact cache if present; fall
        # back to parsing the (slow) filter.json otherwise. Returns None when neither is set.
        compact_cache_path = None
        if self.assets.asset_id or self.repo_id:
            asset_key = self.assets.asset_id or self.repo_id
            compact_cache_path = pathlib.Path(self.assets.assets_dir or assets_dirs) / asset_key / "episodes.json"
        episode_ids = _episode_filter.load_or_resolve(
            abs_repo_ids=repo_ids,
            filter_json_path=self.episode_filter_path,
            compact_cache_path=compact_cache_path,
        )

        repack_transform, data_transforms, model_transforms = _robotwin_transforms(
            model_config,
            use_delta_xyz_actions=self.use_delta_xyz_actions,
            default_prompt=self.default_prompt,
        )
        return dataclasses.replace(
            self.create_base_multi_config(assets_dirs, model_config, repo_ids),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            episode_ids=episode_ids,
        )


@dataclasses.dataclass(frozen=True)
class _LeRobotFrankaDataConfigBase(DataConfigFactory):
    """Base class for Franka real-robot data configs (single-arm and dual-arm).

    Dataset features (from scripts/convert_franka_data_to_lerobot.py):
      Images (always all 4):
        observation.images.{left|right}_{side|wrist}
      State & action — only active arm fields (arm + gripper):
        Single-arm:  observation.state.arm(7,), observation.state.gripper(1,)
                     action.arm(7,), action.gripper(1,)
        Dual-arm:    observation.state.arm(14,), observation.state.gripper(2,)
                     action.arm(14,), action.gripper(2,)
    """

    use_delta_actions: bool = True
    auto_repack: bool = True
    episode_filter_path: str | None = None

    _ALL_STATE_KEYS = frozenset({
        "observation.state.arm",
        "observation.state.gripper",
    })
    _ALL_ACTION_KEYS = frozenset({"action.arm", "action.gripper"})

    def _load_episode_ids(self) -> tuple[int, ...] | None:
        return self._load_episode_ids_from(self.episode_filter_path, self.repo_id)

    @staticmethod
    def _load_episode_ids_from(
        episode_filter_path: str | None, repo_id: str | None
    ) -> tuple[int, ...] | None:
        if not repo_id:
            return None

        if episode_filter_path is not None:
            path = pathlib.Path(episode_filter_path)
        else:
            # Co-located with the dataset ONLY. We deliberately do NOT fall back to a central
            # basename-keyed directory: that borrowed one dataset's filter for a different dataset
            # sharing the task name but a different episode count, producing out-of-range episode
            # indices (e.g. a 168-ep filter applied to a 143-ep dataset -> KeyError 143). A dataset
            # without its own filter.json is trained unfiltered.
            path = pathlib.Path(repo_id) / "filter.json"

        if not path.exists():
            return None

        import json
        text = path.read_text()
        if not text or not text.strip():
            if episode_filter_path is not None:
                logging.info(
                    "episode_filter_path=%s is empty -- treating as no filter (using all episodes).",
                    path,
                )
            return None

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            logging.warning(
                "filter file at %s is not valid JSON (%s) -- using all episodes.",
                path, exc,
            )
            return None

        if not isinstance(raw, dict) or "episodes" not in raw:
            logging.warning("filter.json at %s has no 'episodes' field -- using all episodes.", path)
            return None

        repo_basename = pathlib.Path(repo_id).name
        indices: list[int] = []
        for entry in raw["episodes"]:
            meta = entry.get("metadata") or {}
            entry_repo = meta.get("repo_id", "")
            if entry_repo == repo_id or entry_repo == repo_basename or not entry_repo:
                ep_idx = meta.get("ep_idx")
                if ep_idx is not None:
                    indices.append(int(ep_idx))
        if not indices:
            logging.warning(
                "filter.json at %s matched 0 episodes for repo_id=%s -- using all episodes.",
                path, repo_id,
            )
            return None
        indices = sorted(set(indices))
        logging.info("[FrankaData] Loaded %d episode(s) from filter %s", len(indices), path)
        return tuple(indices)

    def _build_repack_transform(self) -> _transforms.Group:
        """Build repack transform matching the dataset's actual feature keys."""
        if not self.auto_repack:
            return _transforms.Group(
                inputs=[_transforms.RepackTransform(self._default_repack_mapping())]
            )

        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        if repo_id is None:
            return _transforms.Group(
                inputs=[_transforms.RepackTransform(self._default_repack_mapping())]
            )

        try:
            import openpi.training.lerobot_patched as _lerobot_patched
            meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(repo_id)
            feature_keys = set(meta.features.keys())
        except Exception:
            return _transforms.Group(
                inputs=[_transforms.RepackTransform(self._default_repack_mapping())]
            )

        mapping = {}
        mapping["observation.images.left_side"] = "observation.images.left_side"
        mapping["observation.images.left_wrist"] = "observation.images.left_wrist"
        mapping["observation.images.right_side"] = "observation.images.right_side"
        mapping["observation.images.right_wrist"] = "observation.images.right_wrist"

        for key in self._ALL_STATE_KEYS:
            if key in feature_keys:
                mapping[key] = key
        for key in self._ALL_ACTION_KEYS:
            if key in feature_keys:
                mapping[key] = key
        mapping["prompt"] = "prompt"

        arm_shape = meta.features.get("observation.state.arm", {}).get("shape", (7,))
        arm_dim = arm_shape[0] if arm_shape else 7
        mode_msg = "dual-arm" if arm_dim == 14 else f"single-arm (arm={arm_dim})"
        logging.info(f"[FrankaData] Detected dataset mode: {mode_msg}")
        logging.info(f"[FrankaData] Repack mapping keys: {sorted(mapping.keys())}")

        return _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])

    def _default_repack_mapping(self) -> dict:
        return {
            "observation.images.left_side":      "observation.images.left_side",
            "observation.images.left_wrist":     "observation.images.left_wrist",
            "observation.images.right_side":     "observation.images.right_side",
            "observation.images.right_wrist":    "observation.images.right_wrist",
            "observation.state.arm":             "observation.state.arm",
            "observation.state.gripper":         "observation.state.gripper",
            "action.arm":                        "action.arm",
            "action.gripper":                    "action.gripper",
            "prompt":                            "prompt",
        }


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleDataConfig(_LeRobotFrankaDataConfigBase):
    """Franka single-arm (7-dim arm + 1-dim gripper) data config."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_single_policy.FrankaSingleInputs(model_type=model_config.model_type)],
            outputs=[franka_single_policy.FrankaSingleOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualDataConfig(_LeRobotFrankaDataConfigBase):
    """Franka dual-arm (14-dim arm + 2-dim gripper) data config."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_dual_policy.FrankaDualInputs(model_type=model_config.model_type)],
            outputs=[franka_dual_policy.FrankaDualOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


# Backwards-compatible alias.
LeRobotFrankaDataConfig = LeRobotFrankaSingleDataConfig


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleOptimizedDataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized single-arm Franka: 2 cameras, binarized gripper, delta actions."""

    gripper_threshold: float = 120.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleInputs(
                model_type=model_config.model_type,
                gripper_threshold=self.gripper_threshold,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleDeltaDataConfig(_LeRobotFrankaDataConfigBase):
    """Franka single-arm with 3 cameras (left_side + right_side + left_wrist), full delta on joint+gripper."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_single_policy.FrankaSingleDeltaInputs(model_type=model_config.model_type)],
            outputs=[franka_single_policy.FrankaSingleDeltaOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(8)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualDeltaDataConfig(_LeRobotFrankaDataConfigBase):
    """Franka dual-arm with 3 cameras (left_side + left_wrist + right_wrist), full delta on joint+gripper."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_dual_policy.FrankaDualDeltaInputs(model_type=model_config.model_type)],
            outputs=[franka_dual_policy.FrankaDualDeltaOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(16)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualOptimizedDataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized dual-arm Franka: 3 cameras, binarized gripper, delta actions."""

    gripper_threshold: float = 120.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualInputs(
                model_type=model_config.model_type,
                gripper_threshold=self.gripper_threshold,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class _LeRobotFrankaRLDataConfigBase(DataConfigFactory):
    """Base for Franka offline RL data configs (IQL).

    Sets ``extra_delta_timestamps`` so lerobot returns state and all four
    camera streams as a 2-frame stack ``[t, t+H]``, and (when
    ``reward_source == 'progress'``) progress as an ``H+1`` sequence.

    Produces an :class:`RLDataConfig`, which the RL data loader type-checks for.
    """

    use_delta_actions: bool = True
    auto_repack: bool = True
    episode_filter_path: str | None = None
    reward_source: Literal["auto", "progress", "terminal"] = "auto"

    _ALL_STATE_KEYS = frozenset({
        "observation.state.arm",
        "observation.state.gripper",
    })
    _ALL_ACTION_KEYS = frozenset({"action.arm", "action.gripper"})

    def _load_failed_episode_ids(self) -> frozenset[int]:
        """Failed-episode indices co-located with the dataset (``<repo_id>/failed_episodes.json``).

        Offline RL keeps failed episodes in training but applies a terminal failure penalty to
        them (see ``_terminal_bonus_value``), so the reward transform needs the authoritative
        failure labels. Returns an empty set when the file is absent (all episodes then treated
        as non-failed)."""
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        if not repo_id:
            return frozenset()
        path = pathlib.Path(repo_id) / "failed_episodes.json"
        if not path.exists():
            return frozenset()
        import json
        try:
            raw = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001 - defensive: never fail training on a bad label file
            logging.warning("could not parse %s (%s); assuming no failed episodes.", path, e)
            return frozenset()
        out = frozenset(int(i) for i in raw.get("failed_ep_idx", []))
        logging.info("[FrankaRL] Loaded %d failed episode(s) from %s", len(out), path)
        return out

    def _build_rl_config(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
        data_transforms: _transforms.Group,
        delta_action_mask: tuple,
        action_sequence_keys: tuple[str, ...] = ("action.arm", "action.gripper"),
        extra_delta_timestamps: dict[str, tuple[float, ...]] | None = None,
    ) -> RLDataConfig:
        import openpi.training.lerobot_patched as _lerobot_patched

        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        horizon = model_config.action_horizon

        if repo_id:  # non-empty string — load dataset metadata
            meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(repo_id)
            fps = float(meta.fps)
            feature_keys = set(meta.features.keys())
            has_progress = "progress" in feature_keys
        else:
            # Inference-only: no dataset metadata available, use sensible defaults.
            fps = 10.0
            has_progress = False

        next_dt = horizon / fps

        if self.reward_source == "auto":
            resolved_reward: Literal["progress", "terminal"] = "progress" if has_progress else "terminal"
        elif self.reward_source == "progress":
            if not has_progress:
                raise ValueError(
                    f"reward_source='progress' requires a 'progress' feature in {repo_id}, "
                    f"but found only: {sorted(feature_keys)}"
                )
            resolved_reward = "progress"
        else:
            resolved_reward = "terminal"

        extra_dt: dict[str, tuple[float, ...]] = dict(extra_delta_timestamps or {})
        # Defaults for the RL next-step / progress channels. setdefault keeps any
        # keys the caller already registered via action_sequence_keys (which the
        # rl data loader will honor with its own length, e.g. H+1 frames for
        # next-state-as-action) untouched.
        extra_dt.setdefault("observation.images.left_side", (0.0, next_dt))
        extra_dt.setdefault("observation.images.left_wrist", (0.0, next_dt))
        extra_dt.setdefault("observation.images.right_side", (0.0, next_dt))
        extra_dt.setdefault("observation.images.right_wrist", (0.0, next_dt))
        extra_dt.setdefault("observation.state.arm", (0.0, next_dt))
        extra_dt.setdefault("observation.state.gripper", (0.0, next_dt))
        if resolved_reward == "progress":
            extra_dt.setdefault("progress", tuple(t / fps for t in range(horizon + 1)))

        mapping = {}
        for key in [
            "observation.images.left_side",
            "observation.images.left_wrist",
            "observation.images.right_side",
            "observation.images.right_wrist",
        ]:
            mapping[key] = key
        for key in self._ALL_STATE_KEYS:
            if key in feature_keys:
                mapping[key] = key
        for key in self._ALL_ACTION_KEYS:
            if key in feature_keys:
                mapping[key] = key
        mapping["prompt"] = "prompt"
        # Carry the per-sample episode index through repack so the RL reward transform can
        # withhold the terminal bonus for failed episodes (RepackTransform drops unmapped keys).
        mapping["episode_index"] = "episode_index"
        mapping["actions_is_pad"] = "action.arm_is_pad"
        if resolved_reward == "progress":
            mapping["progress"] = "progress"
            if "progress_is_pad" in feature_keys:
                mapping["progress_is_pad"] = "progress_is_pad"
        if "observation.state.arm_is_pad" in feature_keys:
            mapping["observation.state.arm_is_pad"] = "observation.state_arm_is_pad"
        if "observation.state.gripper_is_pad" in feature_keys:
            mapping["observation.state.gripper_is_pad"] = "observation.state_gripper_is_pad"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        # Offline RL keeps ALL episodes (including task failures) in training; failures are
        # handled via reward shaping (no terminal bonus), not by dropping them. Only apply an
        # episode whitelist when one is explicitly provided; never fall back to the central
        # SFT success-filter here.
        episode_ids = (
            _LeRobotFrankaDataConfigBase._load_episode_ids_from(self.episode_filter_path, self.repo_id)
            if self.episode_filter_path is not None
            else None
        )

        base = self.create_base_config(assets_dirs, model_config)
        overrides = {
            "extra_delta_timestamps": extra_dt,
            "repack_transforms": repack_transform,
            "data_transforms": data_transforms,
            "model_transforms": model_transforms,
            "action_sequence_keys": action_sequence_keys,
            "reward_source": resolved_reward,
            "episode_ids": episode_ids,
        }
        carried = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.name not in overrides
        }
        return RLDataConfig(**carried, **overrides)


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleRLDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Franka single-arm offline RL data config."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_single_policy.FrankaSingleRLInputs(model_type=model_config.model_type)],
            outputs=[franka_single_policy.FrankaSingleOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualRLDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Franka dual-arm offline RL data config."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_dual_policy.FrankaDualRLInputs(model_type=model_config.model_type)],
            outputs=[franka_dual_policy.FrankaDualOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1, 7, -1),
        )


# Backwards-compatible alias.
LeRobotFrankaRLDataConfig = LeRobotFrankaSingleRLDataConfig


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleDeltaRLDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Franka single-arm delta RL: 3 cameras (left_side + right_side + left_wrist), full delta."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_single_policy.FrankaSingleDeltaRLInputs(model_type=model_config.model_type)],
            outputs=[franka_single_policy.FrankaSingleDeltaOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(8),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualDeltaRLDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Franka dual-arm delta RL: 3 cameras (left_side + left_wrist + right_wrist), full delta."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_dual_policy.FrankaDualDeltaRLInputs(model_type=model_config.model_type)],
            outputs=[franka_dual_policy.FrankaDualDeltaOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(16),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleRLOptimizedDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized single-arm RL: 2 cameras for policy, 2 for critic, binarized gripper."""

    gripper_threshold: float = 120.0
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleRLInputs(
                model_type=model_config.model_type,
                gripper_threshold=self.gripper_threshold,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualRLOptimizedDataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized dual-arm RL: 3 cameras for policy, 3 for critic, binarized gripper."""

    gripper_threshold: float = 120.0
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualRLInputs(
                model_type=model_config.model_type,
                gripper_threshold=self.gripper_threshold,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputs()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1, 7, -1),
        )



@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleOptimizedV2DataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized single-arm Franka v2: 2 cameras, NO state input, continuous absolute
    joint + gripper actions (no gripper binarization)."""

    use_delta_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleInputsV2(
                model_type=model_config.model_type,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputsV2()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualOptimizedV2DataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized dual-arm Franka v2: 3 cameras, NO state input, continuous absolute
    joint + gripper actions (no gripper binarization)."""

    use_delta_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualInputsV2(
                model_type=model_config.model_type,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputsV2()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleRLOptimizedV2DataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized single-arm RL v2: 2 cameras for policy, 2 for critic, NO state input,
    continuous absolute joint + gripper actions."""

    progress_reward_weight: float = 1.0
    terminal_bonus: float = 1.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleRLInputsV2(
                model_type=model_config.model_type,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputsV2()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualRLOptimizedV2DataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized dual-arm RL v2: 3 cameras for policy, 3 for critic, NO state input,
    continuous absolute joint + gripper actions."""

    progress_reward_weight: float = 1.0
    terminal_bonus: float = 1.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualRLInputsV2(
                model_type=model_config.model_type,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputsV2()],
        )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1, 7, -1),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleOptimizedV3DataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized single-arm Franka v3: 2 cameras, real state (arm+gripper) fed to
    the model, continuous absolute joint + gripper actions."""

    use_delta_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleInputsV3(
                model_type=model_config.model_type,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputsV2()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualOptimizedV3DataConfig(_LeRobotFrankaDataConfigBase):
    """Optimized dual-arm Franka v3: 3 cameras, real state (arm+gripper) fed to
    the model, continuous absolute joint + gripper actions."""

    use_delta_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = self._build_repack_transform()
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualInputsV3(
                model_type=model_config.model_type,
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputsV2()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)

        if self.use_delta_actions:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action.arm", "action.gripper"),
            episode_ids=self._load_episode_ids(),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaSingleRLOptimizedV3DataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized single-arm RL v3: 2 cameras, real state (arm+gripper) fed to the
    model, continuous absolute joint + gripper actions.

    Supports the same ``use_next_state_action`` / ``state_as_input`` flags as
    the SFT v3 config. Note that RL samples transitions ``(s, a, s', r)``;
    ``use_next_state_action`` only affects the action target ``a``, while the
    next state ``s'`` always comes from ``observation.state`` at ``t + H/fps``.
    """

    use_next_state_action: bool = False
    state_as_input: bool = True
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaSingleRLInputsV3(
                model_type=model_config.model_type,
                use_next_state_action=self.use_next_state_action,
                state_as_input=self.state_as_input,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaSingleOutputsV2()],
        )
        # When use_next_state_action is on, fetch H+1 state frames so the
        # transform can slice [1:H+1] as the action chunk; the final frame
        # also doubles as the RL next_state. Pre-scale offsets by 1/fps here
        # because _build_rl_config only applies setdefault to extra_dt and
        # will not override keys already present.
        extra_delta_timestamps: dict[str, tuple[float, ...]] = {}
        if self.use_next_state_action:
            import openpi.training.lerobot_patched as _lerobot_patched
            repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
            meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(repo_id)
            fps = float(meta.fps)
            extra_delta_timestamps["observation.state.arm"] = tuple(
                t / fps for t in range(model_config.action_horizon + 1)
            )
            extra_delta_timestamps["observation.state.gripper"] = tuple(
                t / fps for t in range(model_config.action_horizon + 1)
            )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1),
            extra_delta_timestamps=extra_delta_timestamps,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDualRLOptimizedV3DataConfig(_LeRobotFrankaRLDataConfigBase):
    """Optimized dual-arm RL v3: 3 cameras, real state (arm+gripper) fed to the
    model, continuous absolute joint + gripper actions.

    See :class:`LeRobotFrankaSingleRLOptimizedV3DataConfig` for flag semantics.
    """

    use_next_state_action: bool = False
    state_as_input: bool = True
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> RLDataConfig:
        data_transforms = _transforms.Group(
            inputs=[franka_optimized_policy.OptimizedFrankaDualRLInputsV3(
                model_type=model_config.model_type,
                use_next_state_action=self.use_next_state_action,
                state_as_input=self.state_as_input,
                progress_reward_weight=self.progress_reward_weight,
                terminal_bonus=self.terminal_bonus,
                failed_episodes=self._load_failed_episode_ids(),
            )],
            outputs=[franka_optimized_policy.OptimizedFrankaDualOutputsV2()],
        )
        extra_delta_timestamps: dict[str, tuple[float, ...]] = {}
        if self.use_next_state_action:
            import openpi.training.lerobot_patched as _lerobot_patched
            repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
            meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(repo_id)
            fps = float(meta.fps)
            extra_delta_timestamps["observation.state.arm"] = tuple(
                t / fps for t in range(model_config.action_horizon + 1)
            )
            extra_delta_timestamps["observation.state.gripper"] = tuple(
                t / fps for t in range(model_config.action_horizon + 1)
            )
        return self._build_rl_config(
            assets_dirs, model_config, data_transforms,
            delta_action_mask=_transforms.make_bool_mask(7, -1, 7, -1),
            extra_delta_timestamps=extra_delta_timestamps,
        )

@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class IQLConfig:
    """Hyperparameters for the jaxrl2 PiIQLLearner used by scripts/train_iql.py."""

    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256)
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    latent_dim: int = 50
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    A_scaling: float = 1.0
    critic_reduction: Literal["min", "mean"] = "min"
    encoder_type: str = "resnet_18_v1"
    encoder_norm: str = "group"
    # Color jitter only supports single-camera (3-ch) inputs in jaxrl2; we stack 2 cams
    # into 6 channels for Libero so we keep it off by default. Random-crop still runs.
    color_jitter: bool = False
    use_spatial_softmax: bool = True
    softmax_temperature: float = 1.0
    aug_next: bool = True
    use_bottleneck: bool = True
    num_qs: int = 2
    num_cameras: int = 2
    # Dotted batch-key paths the trainer reads to build the IQL pixel stack.
    # ``image_keys`` are current-view cameras (typically live under
    # ``batch["image"][...]`` after the standard pi0 transforms); the matching
    # ``next_image_keys`` are next-view side-channel keys the RL transform
    # adds (top-level in the batch). Lengths must match ``num_cameras``.
    # Defaults match the LiberoRLInputs 2-camera contract (base + left wrist).
    # RoboTwin uses 3 cameras (base + left/right wrist) and overrides these
    # keys in ``pi05_robotwin_iql``. Adding a camera requires updating both
    # the matching RL transform (so it emits the matching next-view key) and
    # this list.
    image_keys: tuple[str, ...] = ("image.base_0_rgb", "image.left_wrist_0_rgb")
    next_image_keys: tuple[str, ...] = ("next_image_base", "next_image_wrist")
    # Number of critic-warmup steps before the policy starts using IQL advantage weights.
    # During warmup the script behaves like normal flow-matching BC.
    critic_warmup_steps: int = 0


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "/path/to/checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 1000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # IQL hyperparameters. Only consumed by scripts/train_iql.py.
    iql: IQLConfig | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Fine-tuning Franka real-robot configs.
    #
    # SFT (supervised fine-tuning) configs for single-arm and dual-arm Franka.
    # Use --data.repo_id to specify the dataset path on the CLI.
    TrainConfig(
        name="pi05_franka_single",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaSingleDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=6_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_dual",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaDualDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=6_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        save_interval=1_000,
        fsdp_devices=1,
    ),
    #
    # Franka optimized SFT configs.
    # 2/3 cameras, binarized gripper, discrete_state_input=False,
    # 20k steps, lower LR, ema_decay=0.99.
    # Use --data.repo_id to specify the dataset path on the CLI.
    TrainConfig(
        name="pi05_franka_single_optimized",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleOptimizedDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_single_delta",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleDeltaDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_dual_delta",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualDeltaDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_dual_optimized",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualOptimizedDataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    #
    # Franka IQL (offline RL) configs.
    # Requires a dataset with progress or terminal reward signals.
    # Use --data.repo_id to specify the dataset path on the CLI.
    TrainConfig(
        name="pi05_franka_single_iql",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaSingleRLDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=6_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        save_interval=1_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=4,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_side_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_side",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    TrainConfig(
        name="pi05_franka_dual_iql",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaDualRLDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=6_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=10_000,
        save_interval=1_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=4,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_side_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_side",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    #
    # Franka delta IQL configs.
    # 3 cameras, full delta (joint + gripper), discrete_state_input=False.
    TrainConfig(
        name="pi05_franka_single_delta_iql",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleDeltaRLDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.left_side_0_rgb",
                "image.right_side_0_rgb",
                "image.left_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_right_side",
                "next_image_left_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    TrainConfig(
        name="pi05_franka_dual_delta_iql",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualDeltaRLDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    #
    # Franka optimized IQL configs.
    # 2/3 cameras, binarized gripper, discrete_state_input=False,
    # 20k steps, ema_decay=0.99.
    TrainConfig(
        name="pi05_franka_single_iql_optimized",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleRLOptimizedDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=2,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    TrainConfig(
        name="pi05_franka_dual_iql_optimized",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualRLOptimizedDataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    # Franka optimized v2 configs.
    # v2 = 2/3 cameras, NO state input, continuous absolute joint + gripper actions,
    # discrete_state_input=False, 20k steps, ema_decay=0.99.
    TrainConfig(
        name="pi05_franka_single_optimized_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleOptimizedV2DataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_dual_optimized_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualOptimizedV2DataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_single_iql_optimized_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaSingleRLOptimizedV2DataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=2,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    TrainConfig(
        name="pi05_franka_dual_iql_optimized_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
        ),
        data=LeRobotFrankaDualRLOptimizedV2DataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    # Franka optimized v3 configs.
    # v3 = v2 shape but with real joint + gripper state fed to the model
    # (discrete_state_input defaults to True under pi05, so the tokenizer
    # discretizes state into 256 bins and appends it to the language prompt).
    # Actions remain continuous absolute joint + gripper values.
    TrainConfig(
        name="pi05_franka_single_optimized_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaSingleOptimizedV3DataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_dual_optimized_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaDualOptimizedV3DataConfig(
            repo_id="",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            use_delta_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_franka_single_iql_optimized_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaSingleRLOptimizedV3DataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=2,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),
    TrainConfig(
        name="pi05_franka_dual_iql_optimized_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=16,
        ),
        data=LeRobotFrankaDualRLOptimizedV3DataConfig(
            repo_id="",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=False,
            reward_source="auto",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.99,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_interval=2_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.left_side_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_left_side",
                "next_image_left_wrist",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=200,
        ),
    ),


    #
    # Inference Aloha configs.
    #
    #
    # RoboTwin (bimanual EEF) SFT configs.
    #
    # The dataset root contains one lerobot dump per task, e.g.
    #   /path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50/
    #     adjust_bottle-demo_clean_collect_200-50/
    #     beat_block_hammer-demo_clean_collect_200-50/
    #     ...
    # Set ``--data.repo_id=/.../<task_dir>`` on the CLI to fine-tune a single task,
    # or override the repo_id when sweeping. ``prompt_from_task=True`` means the
    # per-episode task string from ``meta/tasks.jsonl`` becomes the language prompt.
    TrainConfig(
        name="pi05_robotwin",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRobotwinDataConfig(
            repo_id="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        save_interval=2_000,
        keep_period=10_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_robotwin_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinDataConfig(
            repo_id="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # LoRA fine-tuning: turn off EMA and freeze non-LoRA params.
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        num_train_steps=30_000,
        save_interval=2_000,
        fsdp_devices=1,
    ),
    #
    # RoboTwin multi-task SFT: fine-tune one model jointly on every sub-task. The factory
    # globs the directory at ``robotwin_root`` and builds a MultiLeRobotDataset over every
    # lerobot dump it finds; the per-row prompt comes from each sub-dataset's own task table.
    # Override ``--data.robotwin_root`` or ``--data.repo_ids`` to use a different set.
    # ``repo_id`` here is a stable identifier used purely for norm-stats / checkpoint pathing.
    #
    TrainConfig(
        name="pi05_robotwin_all",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRobotwinMultiDataConfig(
            repo_id="robotwin_all",
            robotwin_root="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50",
            assets=AssetsConfig(asset_id="robotwin_all"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
            episode_filter_path=ROBOTWIN_EPISODE_FILTER,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        # ~350k frames total across the 50 tasks; train longer than single-task.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        save_interval=1_000,
        keep_period=20_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_robotwin_all_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinMultiDataConfig(
            repo_id="robotwin_all",
            robotwin_root="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50",
            assets=AssetsConfig(asset_id="robotwin_all"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
            episode_filter_path=ROBOTWIN_EPISODE_FILTER,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=1e-4,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        num_train_steps=100_000,
        save_interval=5_000,
        fsdp_devices=1,
    ),
    #
    # RoboTwin augmented (500 ep/task) variant. Same 16-dim EEF schema, 3 cameras, fps=50;
    # the lerobot dump just has ~10x more data per task. Asset_id is distinct so the larger
    # norm-stats file doesn't collide with the clean_50 one.
    #
    TrainConfig(
        name="pi05_robotwin_aug_500",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=False),
        data=LeRobotRobotwinMultiDataConfig(
            repo_id="robotwin_aug_500",
            robotwin_root="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_aug_500",
            assets=AssetsConfig(asset_id="robotwin_aug_500"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
            episode_filter_path=ROBOTWIN_EPISODE_FILTER,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        # ~10x more frames than clean_50; train ~3x longer at the same effective LR.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=300_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=300_000,
        save_interval=10_000,
        keep_period=50_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_robotwin_aug_500_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinMultiDataConfig(
            repo_id="robotwin_aug_500",
            robotwin_root="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_aug_500",
            assets=AssetsConfig(asset_id="robotwin_aug_500"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
            episode_filter_path=ROBOTWIN_EPISODE_FILTER,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=1e-4,
            decay_steps=300_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        num_train_steps=300_000,
        save_interval=10_000,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero_test",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="/path/to/libero_lerobot_data/Libero_PI",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_iql",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoRLDataConfig(
            repo_id="/path/to/libero_lerobot_data/Libero_PI",
            base_config=DataConfig(prompt_from_task=True),
            reward_source="terminal",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=2,
            critic_warmup_steps=2_000,
        ),
    ),
    TrainConfig(
        name="pi05_robotwin_iql",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotRobotwinRLDataConfig(
            repo_id="/path/to/robotwin_progress_data/beat_block_hammer-demo_clean_collect_200-50",
            # Reuse the cross-task RoboTwin norm stats -- the 16-dim EEF state/action
            # space is shared across tasks, so this is a safe out-of-the-box default.
            # Replace with the per-task asset id (and drop assets_dir) if you
            # compute task-specific stats with scripts/compute_norm_stats.py.
            assets=AssetsConfig(assets_dir="./assets/pi05_robotwin_all", asset_id="robotwin_all"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_xyz_actions=False,
            reward_source="terminal",
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=200_000,
        save_interval=1000,
        iql=IQLConfig(
            critic_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile=0.8,
            A_scaling=10.0,
            critic_reduction="min",
            encoder_type="resnet_18_v1",
            color_jitter=False,
            num_cameras=3,
            image_keys=(
                "image.base_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ),
            next_image_keys=(
                "next_image_base",
                "next_image_left_wrist",
                "next_image_right_wrist",
            ),
            critic_warmup_steps=2_000,
        ),
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/path/to/droid_rlds_data",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
