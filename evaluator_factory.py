# evaluator_factory.py
from typing import Optional, Any # Added Any
from evaluator_base import BaseEvaluator
from config import MODEL_CONFIGS

# Import all the specific evaluator classes
from evaluators.qwen2_audio_evaluator import Qwen2AudioEvaluator
from evaluators.kimi_audio_evaluator import KimiAudioEvaluator
from evaluators.audio_flamingo_evaluator import AudioFlamingoEvaluator
from evaluators.omni_r1_evaluator import OmniR1Evaluator



# Modified to accept kwargs
def get_evaluator(model_id: str, **kwargs: Any) -> Optional[BaseEvaluator]:
    """
    Factory function to get the correct evaluator instance based on model_id.
    Passes kwargs (like stepaudio_base_path) to the constructor.
    """
    if model_id not in MODEL_CONFIGS:
        print(f"Error: No configuration found for model '{model_id}' in config.py")
        return None

    config = MODEL_CONFIGS[model_id]
    
    if model_id == "qwen2-audio":
        return Qwen2AudioEvaluator(model_id, config)

    elif model_id == "kimi-audio":
        return KimiAudioEvaluator(model_id, config)

    elif model_id == "omni-r1":
        return OmniR1Evaluator(model_id, config)
    
    elif model_id == "audio-flamingo-3":
        return AudioFlamingoEvaluator(model_id, config)


    else:
        print(f"Error: No evaluator class defined for model '{model_id}' in evaluator_factory.py")
        return None
