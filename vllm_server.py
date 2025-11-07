import os
import time
import wandb
import hydra
import uvicorn
import threading
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.classroom import Classroom, Conversation
from config.train_rl_model import RLModelTrainingConfig
from src.utils.utils import init_logger
import psutil
import torch

logger = init_logger()

import warnings

warnings.filterwarnings("ignore")
load_dotenv()

lock = threading.Lock()

cs = ConfigStore.instance()
cs.store(name="config", node=RLModelTrainingConfig)

classroom: Optional[Classroom] = None
config: Optional[RLModelTrainingConfig] = None
app = FastAPI()

# Readiness state and error tracking
ready_event = threading.Event()
ready_status = {
    "state": "starting",  # starting -> initializing -> ready | error
    "error": None,
}


class ConversationSampleRequest(BaseModel):
    problems: List[str]
    answers: List[str]
    meta: dict = {}


class RewardRequest(BaseModel):
    conversations: list[str]


def _ensure_ready(timeout: int = 1800):
    """Block until classroom is ready or raise 503 after timeout."""
    if ready_event.is_set():
        if ready_status["state"] == "error":
            raise HTTPException(status_code=500, detail=str(ready_status["error"]))
        return
    # Wait up to timeout seconds
    is_ready = ready_event.wait(timeout=timeout)
    if not is_ready:
        raise HTTPException(status_code=503, detail="Server initializing models; try again later.")


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    return {"status": ready_status["state"], "error": str(ready_status["error"]) if ready_status["error"] else None}


@app.get("/metrics")
def metrics():
    """Basic observability: GPU and process memory usage."""
    try:
        gpus = []
        if torch.cuda.is_available():
            num = torch.cuda.device_count()
            for i in range(num):
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()
                gpus.append(
                    {
                        "id": i,
                        "total_bytes": int(total),
                        "free_bytes": int(free),
                        "used_bytes": int(total - free),
                    }
                )
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss
        return {
            "status": ready_status["state"],
            "process_rss_bytes": int(mem),
            "gpus": gpus,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sample_conversations")
def sample_conversations(request: ConversationSampleRequest):
    global classroom, config
    _ensure_ready()

    problems = request.problems
    answers = request.answers
    meta = request.meta
    conversations = None
    with lock:
        conversations = classroom.sample_conversations(
            problems=problems, answers=answers, meta=meta
        )

    df_table = classroom.to_pd_latest()
    rewards = [classroom.get_pedagogical_reward(c) for c in conversations]
    df_table["pedagogical_reward"] = rewards
    df_table["total_reward"] = rewards
    df_table = df_table.astype(str)
    if config.logging.wandb:
        wandb.log(
            {
                f"batch_{len(classroom.conversation_sets)}": wandb.Table(
                    dataframe=df_table
                )
            }
        )

    return [c.get_trainable_representation() for c in conversations]


@app.post("/get_end_rm_reward")
def get_end_rm_reward(request: RewardRequest):
    global classroom
    _ensure_ready()
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_rm_reward(c) for c in conversations]
    return rewards


@app.post("/get_thinking_reward")
def get_thinking_reward(request: RewardRequest):
    global classroom
    _ensure_ready()
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_thinking_reward(c) for c in conversations]
    return rewards


@app.post("/get_end_of_conversation_reward")
def get_end_of_conversation_reward(request: RewardRequest):
    global classroom
    _ensure_ready()
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_end_of_conversation_reward(c) for c in conversations]
    return rewards


@app.post("/get_length_reward")
def get_length_reward(request: RewardRequest):
    global classroom
    _ensure_ready()
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_length_reward(c) for c in conversations]
    return rewards

@app.post("/get_pedagogical_reward")
def get_pedagogical_reward(request: RewardRequest):
    global classroom
    _ensure_ready()
    conversations: list[Conversation] = [
        classroom.get_conversation_by_text(c) for c in request.conversations
    ]
    rewards = [classroom.get_pedagogical_reward(c) for c in conversations]
    return rewards

@app.get("/wait_batch")
def wait_batch():
    # This endpoint waits (blocks) until the current batch (if any) is finished.
    with lock:
        return {"message": "Batch has been run."}


@hydra.main(config_path="config/train_rl", version_base=None)
def _initialize_classroom(cfg: RLModelTrainingConfig):
    """Background thread to initialize Classroom and set readiness state."""
    global classroom, config, ready_status
    try:
        ready_status["state"] = "initializing"
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        classroom = Classroom(
            cfg.student_model,
            cfg.teacher_model,
            cfg.judge_model,
            cfg.reward_model,
            cfg.generation,
            os.path.join(cfg.logging.save_dir, "policy"),
            log_file_path=None,  # hydra_cfg['runtime']['output_dir']
        )
        ready_status["state"] = "ready"
    except Exception as e:
        ready_status["state"] = "error"
        ready_status["error"] = e
    finally:
        ready_event.set()


def main(cfg: RLModelTrainingConfig):
    global classroom, config

    # We merge the config with the defaults
    default_config = OmegaConf.structured(RLModelTrainingConfig)

    # Merge loaded config with defaults
    cfg = OmegaConf.merge(
        default_config, cfg
    )  # Unspecified keys will use defaults from RLModelTrainingConfig

    config = cfg

    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.wandb_project + "-server",
            name=cfg.logging.wandb_run_name,
            entity=cfg.logging.wandb_entity,
            group=cfg.logging.run_group,
            tags=cfg.logging.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )

    # Initialize classroom in the background so the server binds immediately
    t = threading.Thread(target=_initialize_classroom, args=(cfg,), daemon=True)
    t.start()

    # Start the server right away so health/ready endpoints are reachable
    uvicorn.run(app, host="0.0.0.0", port=cfg.generation.server_port)


if __name__ == "__main__":
    main()
