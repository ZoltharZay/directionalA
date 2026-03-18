# evaluators/audio_flamingo_evaluator.py
import time
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional
from transformers import (
    AutoProcessor,
    AudioFlamingo3ForConditionalGeneration
)

from evaluator_base import BaseEvaluator
import audio_utils
import prompt_templates

class AudioFlamingoEvaluator(BaseEvaluator):
    """
    Evaluator for the local nvidia/audio-flamingo-3-chat model.
    """
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        print(f"Loading {config['model_name']} to {self.device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                config["model_name"],
                trust_remote_code=config["trust_remote_code"]
            )
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                config["model_name"],
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=config["trust_remote_code"],
            )
            self.model.eval()
            print("Successfully loaded Audio-Flamingo model.")
        except Exception as e:
            print(f"Error loading Audio-Flamingo: {e}")
            raise

    def _call_model(self, chat: list, audio_waveforms: list) -> str:
        """Helper to call the model's generate function."""
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": chat},
                        {"type": "audio", "path": str(audio_waveforms)},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )

            model_device = self.model.device
            model_dtype = next(self.model.parameters()).dtype

            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    v = v.to(model_device)
                    if torch.is_floating_point(v):
                        v = v.to(model_dtype)
                    inputs[k] = v

            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=1000)

            decoded = self.processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0]
            return decoded.strip()

        except Exception as e:
            return f"Error during Audio-Flamingo generation: {e}"

    def process_audio(
        self, 
        audio_path: str, 
        prompt: str,
        audio_mode: str,
        caption: Optional[str] = None,
        left_caption: Optional[str] = None,
        right_caption: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Process audio with Audio-Flamingo.
        """
        start_time = time.time()
        
        # --- Handle Interactive Modes ---
        if audio_mode == "interactive-single":
            print("Interactive-Single: Generating description...")
            desc_prompt = prompt_templates.DESCRIBE_AUDIO_PROMPT
            # 1. Get raw caption
            raw_caption, _ = self.process_audio(audio_path, desc_prompt, "direct-single")
            # 2. Add prefix
            caption = f"Caption: {raw_caption}" if raw_caption else ""
            # 3. Format prompt
            prompt = prompt.format(caption=caption, left_caption="", right_caption="")

        elif audio_mode == "interactive-double":
            print("Interactive-Double: Generating L/R descriptions...")
            desc_l_prompt = prompt_templates.DESCRIBE_LEFT_AUDIO_PROMPT
            desc_r_prompt = prompt_templates.DESCRIBE_RIGHT_AUDIO_PROMPT
            left_wave, right_wave = audio_utils.get_audio_channels(audio_path)
            
            # 1. Get raw captions
            raw_left_caption, _ = self._process_single_waveform(left_wave, desc_l_prompt)
            raw_right_caption, _ = self._process_single_waveform(right_wave, desc_r_prompt)
            # 2. Add prefixes
            left_caption = f"Left Caption: {raw_left_caption}" if raw_left_caption else ""
            right_caption = f"Right Caption: {raw_right_caption}" if raw_right_caption else ""
            # 3. Format prompt
            prompt = prompt.format(caption="", left_caption=left_caption, right_caption=right_caption)

        # --- Prepare Model Inputs ---
        audio_waveforms = []
        chat = [{"role": "user", "content": ""}]
        
        if audio_mode.endswith("double"):
            left_wave, right_wave = audio_utils.get_audio_channels(audio_path)
            if left_wave is None or right_wave is None:
                return "Error loading L/R audio channels.", 0.0
            audio_waveforms = [left_wave, right_wave]
            chat[0]["content"] = f"<audio> (Left) and <audio> (Right)\n{prompt}"
        else:
            single_wave = audio_utils.get_single_audio_waveform(audio_path)
            if single_wave is None:
                return "Error loading single audio.", 0.0
            audio_waveforms = [single_wave]
            chat[0]["content"] = f"<audio>\n{prompt}"

        response_content = self._call_model(prompt, audio_path)
        processing_time = time.time() - start_time
        
        return response_content, processing_time, prompt
        
    def _process_single_waveform(self, waveform: Optional[np.ndarray], prompt: str) -> Tuple[str, float]:
        """Internal helper for interactive mode."""
        if waveform is None: return "Error processing waveform.", 0.0
        start_time = time.time()
        chat = [{"role": "user", "content": f"<audio>\n{prompt}"}]
        response_content = self._call_model(chat, [waveform])
        processing_time = time.time() - start_time
        return response_content, processing_time