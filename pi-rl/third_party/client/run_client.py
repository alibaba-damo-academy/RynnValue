"""Rollout server for RL training.

Serves as a websocket server to handle environment operations requested by the training server.
Supports operations: create_env, reset, step, get_observation, get_info_for_step.
"""

import asyncio
import dataclasses
import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import websockets
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import tyro

def load_task_config(config_path: Optional[str]):
    """Load task config from module path, similar to config_flags.DEFINE_config_file."""
    if config_path is None:
        return None
    
    # Convert file path to module path if needed (e.g., "configs/task/pick.py" -> "configs.task.pick")
    if '/' in config_path or '.py' in config_path:
        config_path = config_path.replace('.py', '').replace('/', '.')
    
    try:
        module = __import__(config_path, fromlist=['get_config'])
        return module.get_config()
    except Exception as e:
        raise ImportError(f"Failed to load task config from '{config_path}': {e}")


@dataclasses.dataclass
class Args:
    """Configuration arguments for rollout server."""

    server_host: str = "0.0.0.0"
    server_port: int = 8102
    config_task_path: str = "configs/task/franka_single.py"


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_task_config: Optional[Any] = None

# Human-in-the-loop: lazy spacemouse for droid envs
_spacemouse_policy: Optional[Any] = None
_HUMAN_OVERRIDE_NORM_THRESHOLD = 1e-4


def _get_human_override_action(task_config: Optional[Any] = None) -> tuple:
    """Return (action_7d or None, is_human). Assumes 7D action space."""
    global _spacemouse_policy
    try:
        if _spacemouse_policy is None:
            from client.real_utils.spacemouse import SpaceMousePolicy
            _spacemouse_policy = SpaceMousePolicy(
                max_lin_vel=task_config.collect_max_lin_vel,
                max_rot_vel=task_config.collect_max_rot_vel,
            )
        action_7d, _ = _spacemouse_policy.forward(None, include_info=True)
        is_active = np.linalg.norm(action_7d[:6]) > _HUMAN_OVERRIDE_NORM_THRESHOLD
        return (action_7d, True) if is_active else (None, False)
    except Exception as e:
        logging.getLogger(__name__).warning("Spacemouse unavailable (%s), using policy action.", e)
        return None, False


async def _handle_environment_request(websocket: _server.ServerConnection):
    """Handle robomimic operation requests from training server."""
    global _task_config
    logger = logging.getLogger(__name__)
    packer = msgpack_numpy.Packer()
    
    try:
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                operation = request.get("operation")
                
                if operation == "create_env":
                    task_config = load_task_config(_config_task_path)
                    _task_config = task_config
                    env_name = task_config.env_name
                    env_usage = request["env_usage"]
                    env_id = f"{env_name}_{env_usage}"

                    # If an env with the same id already exists (e.g. previous
                    # client crashed without calling close_env), destroy it
                    # first so its background ROS2 spin thread does not collide
                    # with the new one ("generator already executing").
                    old_env = _env_storage.pop(env_id, None)
                    if old_env is not None:
                        logger.info(f"Destroying stale environment {env_id}...")
                        try:
                            old_env.close()
                        except Exception:
                            logger.exception("Error closing stale env %s", env_id)

                    logger.info(f"Creating environment {env_id}...")
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    logger.info(f"Environment {env_id} created successfully")

                    task_description = task_config.language_instruction or getattr(env, "language_instruction", "")
                    response = {"status": "success", "env_id": env_id, "task_description": task_description}
                    await websocket.send(packer.pack(response))
                    logger.info(f"Sent create_env response for {env_id}")
                    
                elif operation == "reset":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.reset()
                        response = {
                            "status": "success",
                            "observation": obs,
                            "done": False,
                        }
                    await websocket.send(packer.pack(response))
                    
                elif operation == "step":
                    env_id = request["env_id"]
                    sent_action = np.array(request["action"])
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        sent_action = sent_action.astype(np.float64)
                        if not np.isfinite(sent_action).all():
                            logger.warning(
                                "Action contains NaN/Inf; replacing with zeros. "
                                "Check policy inputs (observations, encoder), training stability, or checkpoint."
                            )
                            sent_action = np.where(np.isfinite(sent_action), sent_action, 0.0)
                        real_action = sent_action.copy()
                        action_type = "policy"
                        is_human = False
                        if _task_config is not None and _task_config.env_type == "droid" and getattr(_task_config, 'enable_hil', True):
                            sm_action, is_human = _get_human_override_action(_task_config)
                            if is_human and sm_action is not None:
                                real_action[:6] = sm_action[:6]
                                real_action[6] = sm_action[6]
                                action_type = "human"
                        sent_is_invalid = np.allclose(sent_action, -1.0)
                        if is_human or not sent_is_invalid:
                            step_result = env.step(real_action)
                            executed_action = np.array(
                                step_result["executed_action"],
                                dtype=np.float64,
                            )
                        else:
                            executed_action = real_action

                        response = {
                            "status": "success",
                            "action": executed_action.tolist(),
                            "action_type": action_type,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "step_chunk":
                    env_id = request["env_id"]
                    sent_chunk = np.array(request["action_chunk"], dtype=np.float64)
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    elif not hasattr(env, "send_action_chunk"):
                        response = {"status": "error", "message": f"Environment {env_id} does not support send_action_chunk"}
                    else:
                        if not np.isfinite(sent_chunk).all():
                            logger.warning(
                                "Action chunk contains NaN/Inf; replacing invalid values with zeros."
                            )
                            sent_chunk = np.where(np.isfinite(sent_chunk), sent_chunk, 0.0)

                        chunk_result = env.send_action_chunk(
                            sent_chunk,
                            action_index=request.get("action_index"),
                            execute_steps=request.get("execute_steps"),
                            publish_hz=request.get("publish_hz"),
                            return_observations=True,
                        )
                        observations = chunk_result.get("observations") or [env.get_observation()]
                        dones = chunk_result.get("dones") or []
                        successes = chunk_result.get("successes") or []
                        rewards = chunk_result.get("rewards") or []
                        masks = chunk_result.get("masks") or []
                        if dones:
                            done = bool(dones[-1])
                            success = bool(successes[-1])
                            reward = float(rewards[-1])
                            mask = float(masks[-1])
                        else:
                            done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "actions": np.asarray(
                                chunk_result["executed_actions"],
                                dtype=np.float64,
                            ).tolist(),
                            "observations": observations,
                            "dones": [bool(v) for v in dones],
                            "successes": [bool(v) for v in successes],
                            "rewards": [float(v) for v in rewards],
                            "masks": [float(v) for v in masks],
                            "observation": observations[-1],
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                            "action_type": "policy",
                            "publish_status": chunk_result.get("publish_status", {}),
                            "start_idx": int(chunk_result.get("start_idx", 0)),
                            "end_idx": int(chunk_result.get("end_idx", 0)),
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_observation":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.get_observation()
                        response = {
                            "status": "success",
                            "observation": obs,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_info_for_step":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)
                    
                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "close_env":
                    env_id = request["env_id"]
                    env = _env_storage.pop(env_id, None)
                    if env is None:
                        response = {"status": "success", "message": f"Environment {env_id} not found (already closed?)"}
                    else:
                        try:
                            env.close()
                            logger.info(f"Closed environment {env_id}")
                            response = {"status": "success"}
                        except Exception as e:
                            logger.exception("Error closing env %s", env_id)
                            response = {"status": "error", "message": f"Failed to close {env_id}: {e}"}
                    await websocket.send(packer.pack(response))

                else:
                    response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    await websocket.send(packer.pack(response))
            
            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Connection closed by client {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                try:
                    response = {"status": "error", "message": str(e)}
                    await websocket.send(packer.pack(response))
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Connection closed while sending error response")
                    break
                
    except websockets.exceptions.ConnectionClosed:
        logger.debug(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Unexpected error in request handler: {e}", exc_info=True)


async def _run_server(host: str, port: int, config_task_path: Optional[str]):
    """Run the websocket server for environment operations."""
    global _config_task_path
    _config_task_path = config_task_path
    logger = logging.getLogger(__name__)

    async with _server.serve(
        _handle_environment_request, 
        host, 
        port, 
        compression=None, 
        max_size=None,
        # This server handles potentially long blocking work (env init/step).
        # Disable keepalive pings to avoid ping timeouts while busy.
        ping_interval=None,
        ping_timeout=None,
        close_timeout=100,
    ) as server:
        logger.info(f"Environment operations server started on {host}:{port}")
        await server.serve_forever()

async def main_async(args: Args) -> None:
    """Main async entry point."""
    await _run_server(args.server_host, args.server_port, args.config_task_path)

def main(args: Args) -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
