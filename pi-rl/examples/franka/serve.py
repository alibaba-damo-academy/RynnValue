# adapted from openpi
import dataclasses
import logging
from contextlib import asynccontextmanager

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoModel, AutoProcessor

from open_server_client.policy.damo import DamoVLAPolicy
from open_server_client.policy.recoder import PolicyRecorder
from pb_utils import ObservationReader


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    port: int = 8123
    record: bool = False
    model_path: str = ""
    token: str = ""
    num_workers: int = 1
    action_rate: int = 30
    pack: str = "protobuf"  # msgpack or protobuf
    env: str = "LEROBOT"
    compress: str = "gzip"
    angle_type: str = "degree"
    num_steps: int = 10
    gripper_limit: float = 100.0


class VLAPolicyServerApp:
    def __init__(self, args: Args):
        self.args = args
        self.model = None
        self.processor = None
        self.obs_reader = None
        self.policy = None
        self.server = None

        self.app = FastAPI(lifespan=self.lifespan)
        self._register_routes()

    async def lifespan(self, app: FastAPI):
        logging.basicConfig(level=logging.INFO, force=True)
        self._initialize_components(app)
        yield
        self._shutdown()

    def _initialize_components(self, app: FastAPI):
        logging.info("Initializing model and server components...")

        self.model = AutoModel.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.cuda()

        self.processor = AutoProcessor.from_pretrained(self.args.model_path)
        self.obs_reader = ObservationReader(
            env=self.args.env.lower(),
            angle_type=self.args.angle_type,
        )

        self.policy = DamoVLAPolicy(eval_func=self.server_get_vla_action)

        if self.args.record:
            self.policy = PolicyRecorder(self.policy, "policy_records")

        if self.args.pack == "msgpack":
            from open_server_client.http_server_policy import HttpPolicyServer
        elif self.args.pack == "protobuf":
            from open_server_client.http_server_policy_protoc_robot import HttpPolicyServer
        else:
            raise NotImplementedError(f"args.pack={self.args.pack} type not supported")

        self.server = HttpPolicyServer(
            policy=self.policy,
            host="0.0.0.0",
            port=self.args.port,
            token=self.args.token,
            metadata=self.policy.metadata,
            num_workers=self.args.num_workers,
            env=self.args.env.lower(),
            action_rate=self.args.action_rate,
            compress=self.args.compress,
        )

        allocated_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"Currently allocated GPU memory: {allocated_memory_gb:.2f} GB")

        app.state.model_path = self.args.model_path
        app.state.model_config = self.model.config.to_dict()

        logging.info("Initialization complete.")

    def _shutdown(self):
        logging.info("Shutting down server...")
        self.server = None

    def _register_routes(self):
        @self.app.post("/")
        async def root(request: Request):
            return await self.server.root(request)

        @self.app.get("/info")
        async def info():
            return {
                "model_path": self.app.state.model_path,
                "model_config": self.app.state.model_config,
            }

    def _extract_cameras(self, obs: dict) -> dict:
        cameras = {}

        if self.args.env.lower() == "realman":
            key_mapping = {
                "observation/image": "observation.images.head_camera",
                "observation/left_wrist_image": "observation.images.left_hand_camera",
                "observation/right_wrist_image": "observation.images.right_hand_camera",
                "observation.image": "observation.images.head_camera",
                "observation.left_wrist_image": "observation.images.left_hand_camera",
                "observation.right_wrist_image": "observation.images.right_hand_camera",
            }

            for client_key, standard_key in key_mapping.items():
                if client_key in obs:
                    try:
                        cameras[standard_key] = Image.fromarray(obs[client_key]).convert("RGB")
                        logging.info(
                            f"  ✅ Mapped '{client_key}' -> '{standard_key}', "
                            f"shape={obs[client_key].shape}"
                        )
                    except Exception as e:
                        logging.error(f"  ❌ Failed to convert '{client_key}' to image: {e}")
                else:
                    logging.warning(f"  ⚠️ Client key '{client_key}' NOT FOUND in observation")
        else:
            logging.info("🔍 Checking for standard format images (observation.images.*)...")
            for k, v in obs.items():
                if k.startswith("observation.images."):
                    try:
                        cameras[k] = Image.fromarray(v).convert("RGB")
                        logging.info(f"  ✅ Found standard format image: '{k}', shape={v.shape}")
                    except Exception as e:
                        logging.error(f"  ❌ Failed to convert '{k}' to image: {e}")

        return cameras

    @torch.inference_mode()
    def server_get_vla_action(self, obs: dict) -> dict:
        logging.info("=" * 80)
        logging.info("🚀 NEW REQUEST RECEIVED")
        logging.info("=" * 80)

        logging.info(f"📦 Observation keys received: {list(obs.keys())}")
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                logging.info(f"  - '{k}': shape={v.shape}, dtype={v.dtype}")
            else:
                logging.info(f"  - '{k}': type={type(v)}")

        cameras = self._extract_cameras(obs)

        if not cameras:
            logging.error("❌ NO CAMERAS EXTRACTED! Model will run without visual input!")
        else:
            logging.info(f"✅ Extracted {len(cameras)} cameras: {list(cameras.keys())}")

        try:
            prompt = self.obs_reader.get_prompt(obs)
            logging.info(f"📝 Instruction/Prompt: '{prompt}'")
        except Exception as e:
            logging.error(f"❌ Failed to extract prompt: {e}")
            prompt = ""

        try:
            state = self.obs_reader.get_state(obs)
            logging.info(f"🔢 State extracted: shape={state.shape}, dtype={state.dtype}")
            logging.info(f"   State values (first 5): {state[:5]}")
        except Exception as e:
            logging.error(f"❌ Failed to extract state: {e}")
            logging.error(f"   Available keys for state: {[k for k in obs.keys() if 'state' in k.lower()]}")
            raise

        logging.info("🔄 Building conversation with processor...")
        conversation = self.processor.build_conversation(
            instruction=prompt,
            cameras=cameras,
            states=[state.tolist()],
        )
        logging.info("✅ Conversation built successfully")

        model_inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        logging.info(f"🔧 Model inputs prepared: {list(model_inputs.keys())}")
        for k, v in model_inputs.items():
            if hasattr(v, "shape"):
                logging.info(f"   - {k}: shape={v.shape}")

        model_inputs = model_inputs.to(self.model.device)

        logging.info(f"🤖 Running model inference with num_steps={self.args.num_steps}...")
        try:
            actions = self.model.sample(
                **model_inputs,
                num_steps=self.args.num_steps,
            ).cpu().numpy()[0]
            logging.info(f"✅ Model inference completed, actions shape: {actions.shape}")
            logging.info(f"   Raw actions (first 3 steps):\n{actions[:3]}")
        except Exception as e:
            logging.error(f"❌ Model inference FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

        original_gripper = actions[:, -1].copy()
        actions[:, -1] = np.minimum(actions[:, -1], self.args.gripper_limit)
        if not np.allclose(original_gripper, actions[:, -1]):
            logging.info(f"⚙️ Gripper values clamped to limit={self.args.gripper_limit}")
            logging.info(f"   Before: {original_gripper[:3]}")
            logging.info(f"   After: {actions[:, -1][:3]}")

        logging.info("🔄 Formatting actions for output...")
        outputs = self.obs_reader.format_action(actions)
        logging.info(f"✅ Actions formatted: {list(outputs.keys())}")
        for k, v in outputs.items():
            if isinstance(v, np.ndarray):
                logging.info(f"   - '{k}': shape={v.shape}")
                logging.info(f"      First 3 steps:\n{v[:3]}")

        logging.info("=" * 80)
        logging.info("✅ REQUEST COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)

        return outputs

    def run(self):
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.args.port,
            workers=self.args.num_workers,
            loop="auto",
            http="httptools",
        )


def main():
    args = tyro.cli(Args)
    server_app = VLAPolicyServerApp(args)
    server_app.run()


if __name__ == "__main__":
    main()
