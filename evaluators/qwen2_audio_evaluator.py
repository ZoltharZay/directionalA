# evaluators/qwen2_audio_evaluator.py
import time
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)

from evaluator_base import BaseEvaluator
from transformers.pipelines.audio_utils import ffmpeg_read
import audio_utils
import prompt_templates
import requests

class Qwen2AudioEvaluator(BaseEvaluator):
    """
    Evaluator for the local Qwen2-Audio Hugging Face model.
    FORCES MONO on single inputs to resolve padding error.
    Corrected indentation and simplified generate call.
    """
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading {config['model_name']} to {self.device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                config["model_name"],
                trust_remote_code=config["trust_remote_code"]
            )
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                config["model_name"],
                dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=config["trust_remote_code"]
            )
            print("Successfully loaded Qwen2-Audio model.")
        except Exception as e:
            print(f"Error loading Qwen2-Audio: {e}")
            raise


    def read_audio(self, audio_path):
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(audio_path).content
        else:
            with open(audio_path, "rb") as f:
                inputs = f.read()
        return inputs

    def _call_qwen_model(self, text: str, audio_list: list) -> str:
        """Helper to call the model."""
        try:
            # Workaround for Padding Error
            input_audios = ffmpeg_read(self.read_audio(audio_list), sampling_rate=self.processor.feature_extractor.sampling_rate) 

            # Process inputs
            inputs = self.processor(
                text=text,
                audio=input_audios,
                sampling_rate=audio_utils.TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(self.device, dtype=self.torch_dtype)

            # Define Generation Config
            pad_token_id = getattr(self.processor, 'tokenizer', self.processor).pad_token_id
            eos_token_id = getattr(self.processor, 'tokenizer', self.processor).eos_token_id
            generation_config = GenerationConfig(
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )

            # --- Prepare arguments for model.generate ---
            # Pass the dictionary returned by the processor directly.
            # It contains input_ids, attention_mask, input_features, etc.
            generate_kwargs = inputs.copy() # Use all keys from processor output
            generate_kwargs["generation_config"] = generation_config

            # --- THIS BLOCK IS WHERE THE INDENTATION ERROR WAS ---
            # Simplified the call below, removing the problematic explicit assignment.

            with torch.no_grad():
                # Pass the whole dictionary using **
                generated_ids = self.model.generate(**generate_kwargs)

            prompt_length = inputs["input_ids"].size(1)
            response_ids = generated_ids[:, prompt_length:]

            generated_text = self.processor.batch_decode(
                response_ids, skip_special_tokens=True
            )[0]

            parts = generated_text.split("assistant\n")
            if len(parts) > 1:
                return parts[-1].strip()
            else:
                 # Try to clean based on known template structure
                 templated_prompt_end = "<|im_start|>assistant\n"
                 # Find the end of the prompt part in the input text
                 prompt_part_end_index = text.find(templated_prompt_end)
                 if prompt_part_end_index != -1:
                      # Extract the actual prompt text passed to the processor
                      prompt_text_passed = text[:prompt_part_end_index + len(templated_prompt_end)]
                      # Remove this exact text from the beginning of the output
                      if generated_text.startswith(prompt_text_passed):
                           cleaned_text = generated_text[len(prompt_text_passed):].strip()
                           return cleaned_text if cleaned_text else generated_text
                 # Fallback cleaning (less reliable)
                 cleaned_text = generated_text.replace(text, "").strip()
                 return cleaned_text if cleaned_text else generated_text

        except Exception as e:
            import traceback
            print(f"Error during Qwen model generation: {e}")
            traceback.print_exc()
            return f"Error during Qwen model generation: {e}"

    # --- process_audio and helper methods remain unchanged ---

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
        Process audio with Qwen2-Audio, handling all modes.
        """
        start_time = time.time()
        final_prompt = prompt # Start with the base prompt

        # --- Handle Interactive Modes ---
        if audio_mode == "interactive-single":
            print("Interactive-Single: Generating description...")
            desc_prompt_text = prompt_templates.DESCRIBE_AUDIO_PROMPT
            raw_caption, _ = self._process_audio_core(audio_path, desc_prompt_text, "direct-single")
            caption = f" Audio Caption: {raw_caption}\n" if raw_caption else ""
            final_prompt = caption + prompt

        elif audio_mode == "interactive-double":
            print("Interactive-Double: Generating L/R descriptions...")
            desc_l_prompt_text = prompt_templates.DESCRIBE_LEFT_AUDIO_PROMPT
            desc_r_prompt_text = prompt_templates.DESCRIBE_RIGHT_AUDIO_PROMPT
            left_wave, right_wave = audio_utils.get_audio_channels(audio_path)

            raw_left_caption, _ = self._process_single_waveform(left_wave, desc_l_prompt_text)
            raw_right_caption, _ = self._process_single_waveform(right_wave, desc_r_prompt_text)
            left_caption= f"Left Audio Channal Caption: {raw_left_caption}\n" if raw_left_caption else ""
            right_caption = f"Right Audio Channal Caption: {raw_right_caption}\n" if raw_right_caption else ""
            final_prompt = left_caption + right_caption + prompt
        else:
            final_prompt = prompt.format(caption="", left_caption="", right_caption="")

        # --- Call Core Logic ---
        response_content, _ = self._process_audio_core(audio_path, final_prompt, audio_mode)

        processing_time = time.time() - start_time
        return response_content, processing_time, final_prompt


    def _process_audio_core(self, audio_path: str, prompt_text: str, audio_mode: str) -> Tuple[str, float]:
         """Core logic shared by process_audio and interactive caption generation."""
         start_time = time.time()
         audio_list = []
         conversation_struct = []

         if audio_mode.endswith("double"):
             left_wave, right_wave = audio_utils.get_audio_channels(audio_path)
             if left_wave is None or right_wave is None:
                 return "Error loading L/R audio channels.", 0.0
             audio_list = [left_wave, right_wave]
             conversation_struct = [
                 {'role': 'system', 'content': 'You are a helpful assistant. Check both Audio1 (Left) and Audio2 (Right).'},
                 {"role": "user", "content": [ {"type": "audio"}, {"type": "audio"}, {"type": "text", "text": prompt_text} ]},
             ]
         else: # Handles direct-single and the first step of interactive-single
             single_wave = audio_utils.get_single_audio_waveform(audio_path)
             if single_wave is None:
                 return "Error loading single audio.", 0.0
             audio_list = [single_wave]
             conversation_struct = [
                 {'role': 'system', 'content': 'You are a helpful assistant.'},
                 {"role": "user", "content": [ {"type": "audio"}, {"type": "text", "text": prompt_text} ]},
             ]

         # Apply chat template *before* calling the model helper
         templated_text = self.processor.apply_chat_template(
             conversation_struct, add_generation_prompt=True, tokenize=False
         )
         print("Audio Path: " + audio_path)
         response_content = self._call_qwen_model(templated_text, audio_path)
         processing_time = time.time() - start_time
         return response_content, processing_time

    def _process_single_waveform(self, waveform: Optional[np.ndarray], prompt_text: str) -> Tuple[str, float]:
        """Internal helper for interactive-double mode to process a raw waveform."""
        if waveform is None:
            return "Error processing waveform.", 0.0

        start_time = time.time()
        audio_list = [waveform] # Waveform here is already mono from get_audio_channels
        conversation_struct = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [ {"type": "audio"}, {"type": "text", "text": prompt_text} ]},
        ]
        templated_text = self.processor.apply_chat_template(
            conversation_struct, add_generation_prompt=True, tokenize=False
        )

        response_content = self._call_qwen_model(templated_text, audio_list)
        processing_time = time.time() - start_time
        return response_content, processing_time
