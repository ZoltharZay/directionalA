# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()





# --- Model Evaluator Configurations ---
MODEL_CONFIGS = {
    "audio-flamingo-3": {
        "type": "local",
        "model_name": "nvidia/audio-flamingo-3-hf",
        "trust_remote_code": True,
    },
    "kimi-audio": {
        "type": "local",
        "model_name": "moonshotai/Kimi-Audio-7B-Instruct",
        "trust_remote_code": True,
    },
    "qwen2-audio": {
        "type": "local",
        "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
        "trust_remote_code": True,
    },
    "qwen-audio": {
        "type": "local",
        "model_name": "Qwen/Qwen-Audio-Chat",
        "trust_remote_code": True,
    },
    "omni-r1": {
        "type": "local",
        "model_name": "Haoz0206/Omni-R1",
        "trust_remote_code": True,
    },
}

