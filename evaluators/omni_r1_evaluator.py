# evaluators/omni_r1_evaluator.py
import time
import torch
import tempfile
import soundfile as sf
from typing import Dict, Tuple, Any, Optional
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration, # From user snippet
    Qwen2_5OmniProcessor, # From user snippet
    GenerationConfig
)
from qwen_omni_utils import process_mm_info # From user snippet

from evaluator_base import BaseEvaluator
import audio_utils
import prompt_templates

# Recommended Qwen system prompt
QWEN_SYS_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."


class OmniR1Evaluator(BaseEvaluator):
    """
    Evaluator for the local Omni-R1 (Haoz0206/Omni-R1) model.
    """
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading {config['model_name']} to {self.device}...")
        try:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                config["model_name"],
                device_map="auto",
                torch_dtype=self.torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=config["trust_remote_code"]
            ).eval()
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                config["model_name"],
                trust_remote_code=config["trust_remote_code"]
            )

            # Use model's default generation config but allow overriding max_new_tokens
            # Get default config and update it
            self.base_generation_config = self.model.generation_config
            self.base_generation_config.max_new_tokens = 256 # Set our desired length
            self.base_generation_config.do_sample = False # Ensure deterministic output

            print("Successfully loaded Omni-R1 model.")
        except Exception as e:
            print(f"Error loading Omni-R1: {e}")
            raise

    def _call_model(self, conversation: list) -> str:
        """Helper to call the model using the user snippet's logic."""
        try:
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Process audio (and empty images/videos)
            audio_input, image_input, video_input = process_mm_info(
                conversation, use_audio_in_video=False
            )

            # === THIS IS THE FIX for the warning ===
            # The processor expects 'audio=', not 'audios='
            inputs = self.processor(
                text=text,
                images=image_input,
                audio=audio_input, # Changed from audios=
                videos=video_input,
                return_tensors="pt",
                do_resize=True, # Keep this? Snippet had it.
            ).to(self.model.device).to(self.torch_dtype)
            # =======================================

            # Generate output
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, generation_config=self.base_generation_config # Use updated config
                )

            prompt_length = inputs["input_ids"].size(1)
            completion_ids = generated_ids[:, prompt_length:]
            text_output = self.processor.batch_decode(completion_ids, skip_special_tokens=True)

            return text_output[0].strip()

        except Exception as e:
            import traceback
            print(f"Error during Omni-R1 generation: {e}")
            traceback.print_exc()
            return f"Error during Omni-R1 generation: {e}"

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
        Process audio with Omni-R1.
        """
        start_time = time.time()
        final_prompt_text = prompt # Start with base prompt template

# --- Handle Interactive Modes ---
        final_prompt_text = prompt # Start with the template string passed in

        if audio_mode == "interactive-single":
            print("Interactive-Single: Generating description...")
            desc_prompt_text = prompt_templates.DESCRIBE_AUDIO_PROMPT
            # 1. Get raw caption using the CORE logic (pass template, mode='direct-single')
            #    Need a way to call the core logic without infinite recursion.
            #    Let's assume a helper _process_audio_core exists or is the main logic.
            #    For simplicity, we'll call a dedicated helper for caption generation.
            raw_caption, _ = self._get_interactive_caption(audio_path, desc_prompt_text, "single")
            # 2. Add prefix if caption exists
            caption_text = f"Audio Caption: {raw_caption}\n" if raw_caption else ""
            # 3. Format the *original template string* with the new caption
            final_prompt_text = prompt.format(caption=caption_text, left_caption="", right_caption="")

        elif audio_mode == "interactive-double":
            print("Interactive-Double: Generating L/R descriptions...")
            desc_l_prompt_text = prompt_templates.DESCRIBE_LEFT_AUDIO_PROMPT
            desc_r_prompt_text = prompt_templates.DESCRIBE_RIGHT_AUDIO_PROMPT

            # 1. Get raw captions for L/R channels
            raw_left_caption, _ = self._get_interactive_caption(audio_path, desc_l_prompt_text, "left")
            raw_right_caption, _ = self._get_interactive_caption(audio_path, desc_r_prompt_text, "right")

            # 2. Add prefixes if captions exist
            left_caption_text = f"Left Audio Channal Caption: {raw_left_caption}\n" if raw_left_caption else ""
            right_caption_text = f"Right Audio Channal Caption: {raw_right_caption}\n" if raw_right_caption else ""
            # 3. Format the *original template string* with the new captions
            final_prompt_text = prompt.format(caption="", left_caption=left_caption_text, right_caption=right_caption_text)
        else:
            final_prompt_text = prompt.format(caption="", left_caption="", right_caption="")

        # --- Call Core Logic ---
        # Pass the final_prompt_text (which is either the original template or the interactively formatted one)
        # and the original audio_path/audio_mode.

        

        # --- Call Core Logic ---
        response_content, _ = self._process_audio_core(audio_path, final_prompt_text, audio_mode)

        processing_time = time.time() - start_time
        return response_content, processing_time, final_prompt_text


    def _process_audio_core(self, audio_path: str, prompt_text: str, audio_mode: str) -> Tuple[str, float]:
        """Core logic using paths, handles single/double modes."""
        start_time = time.time()
        # Use the recommended Qwen system prompt
        messages = [{"role": "system", "content": [{"type": "text", "text": QWEN_SYS_PROMPT}]}]
        user_content = []

        if audio_mode.endswith("double"):
            # Use temp files for L/R channels
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_l, \
                 tempfile.NamedTemporaryFile(suffix=".wav") as tmp_r:

                left_wave, right_wave = audio_utils.get_audio_channels(audio_path)
                sf.write(tmp_l.name, left_wave, audio_utils.TARGET_SAMPLE_RATE)
                sf.write(tmp_r.name, right_wave, audio_utils.TARGET_SAMPLE_RATE)

                user_content.append({"type": "audio", "audio": tmp_l.name})
                user_content.append({"type": "audio", "audio": tmp_r.name})
                user_content.append({"type": "text", "text": prompt_text})
                messages.append({"role": "user", "content": user_content})

                response_content = self._call_model(messages)
        else:
            # Single mode: Pass the original audio path
            user_content.append({"type": "audio", "audio": audio_path})
            user_content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": user_content})

            response_content = self._call_model(messages)

        processing_time = time.time() - start_time
        return response_content, processing_time

    def _process_single_path(self, path: str, prompt_text: str) -> Tuple[str, float]:
        """Internal helper for interactive mode to process a single file path."""
        start_time = time.time()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": QWEN_SYS_PROMPT}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": path},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        response_content = self._call_model(messages)
        processing_time = time.time() - start_time
        return response_content, processing_time
    
    # --- NEW HELPER METHOD ---
    def _get_interactive_caption(self, audio_path: str, desc_prompt: str, channel: str) -> Tuple[str, float]:
        """Helper for interactive modes. Omni-R1 uses paths."""
        start_time = time.time()
        if channel == "left":
            # Need temp file for left channel path
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_l:
                left_wave, _ = audio_utils.get_audio_channels(audio_path)
                if left_wave is None: return "Error loading left channel", 0.0
                sf.write(tmp_l.name, left_wave, audio_utils.TARGET_SAMPLE_RATE)
                # Call core logic with temp path, forcing direct-single mode
                caption, _ = self._process_audio_core(tmp_l.name, desc_prompt, "direct-single")
        elif channel == "right":
            # Need temp file for right channel path
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_r:
                _, right_wave = audio_utils.get_audio_channels(audio_path)
                if right_wave is None: return "Error loading right channel", 0.0
                sf.write(tmp_r.name, right_wave, audio_utils.TARGET_SAMPLE_RATE)
                # Call core logic with temp path, forcing direct-single mode
                caption, _ = self._process_audio_core(tmp_r.name, desc_prompt, "direct-single")
        else: # channel == "single"
            # Use core logic with original path, forcing direct-single mode
            caption, _ = self._process_audio_core(audio_path, desc_prompt, "direct-single")
        processing_time = time.time() - start_time
        return caption, processing_time
    # --- END NEW HELPER ---