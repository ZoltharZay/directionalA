# evaluators/kimi_audio_evaluator.py
import time
import torch
import soundfile as sf
from typing import Dict, Tuple, Any, Optional
import tempfile # Needed for interactive double?
import io
from evaluator_base import BaseEvaluator
import audio_utils
import prompt_templates

# Try to import from the git package
try:
    from kimia_infer.api.kimia import KimiAudio
except ImportError:
    print("ERROR: KimiAudio package not found.")
    print("Please install it by running:")
    print("pip install git+https://github.com/MoonshotAI/Kimi-Audio.git")
    KimiAudio = None

class KimiAudioEvaluator(BaseEvaluator):
    """
    Evaluator for the local Kimi-Audio model.
    """
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        if KimiAudio is None:
            raise ImportError("KimiAudio package is not installed.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {config['model_name']}...") # Kimi handles device internally?

        try:
            self.model = KimiAudio(
                model_path=config["model_name"],
                load_detokenizer=True
            )
            # Sampling params from user snippet
            self.sampling_params = {
                "audio_temperature": 0.8,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                 # Added from other user snippet
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
            }
            print("Successfully loaded Kimi-Audio model.")
        except Exception as e:
            print(f"Error loading Kimi-Audio: {e}")
            raise

    def _call_kimi_model(self, messages: list) -> str:
        """Helper to call the model's generate function."""
        try:
            _, text = self.model.generate(
                messages,
                **self.sampling_params,
                output_type="text"
            )
            return text
        except Exception as e:
            import traceback
            print(f"Error during Kimi model generation: {e}")
            traceback.print_exc()
            return f"Error during Kimi model generation: {e}"

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
        Process audio with Kimi-Audio.
        It expects a file path.
        """
        start_time = time.time()
        final_prompt_text = prompt # Start with base template

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
            print("raw Caption: " + raw_caption)
            # 2. Add prefix if caption exists
            caption_text = f"Audio Caption: {raw_caption}\n" if raw_caption else ""
            print("Caption Text: " + raw_caption)
            # 3. Format the *original template string* with the new caption
            final_prompt_text = prompt.format(caption=caption_text, left_caption="", right_caption="")
            print("Final Prompt Text: " + final_prompt_text)

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
        # Always use the original audio_path for Kimi, even in double mode simulation
        response_content, _ = self._process_audio_core(audio_path, final_prompt_text, audio_mode)

        processing_time = time.time() - start_time
        return response_content, processing_time, final_prompt_text


    def _process_audio_core(self, audio_path: str, prompt_text: str, audio_mode: str) -> Tuple[str, float]:
        start_time = time.time()
        if audio_mode.endswith("doubles"):
            left_wave, right_wave = audio_utils.get_audio_channels(audio_path)

            # Create both temp files in a single context
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_l, tempfile.NamedTemporaryFile(suffix=".wav") as tmp_r:
                sf.write(tmp_l.name, left_wave, audio_utils.TARGET_SAMPLE_RATE)
                sf.write(tmp_r.name, right_wave, audio_utils.TARGET_SAMPLE_RATE)

                # Directly build the messages while temp files are still alive
                messages = [
                    {"role": "user", "message_type": "text", "content": prompt_text},
                    {"role": "user", "message_type": "audio", "content": tmp_l.name},
                    {"role": "user", "message_type": "audio", "content": tmp_r.name},
                ]
                response_content = self._call_kimi_model(messages)
                processing_time = time.time() - start_time
            return response_content, processing_time
        else:        
            messages = [
             {"role": "user", "message_type": "text", "content": prompt_text},
             {"role": "user", "message_type": "audio", "content": audio_path}
         ]
            response_content = self._call_kimi_model(messages)
            processing_time = time.time() - start_time
            return response_content, processing_time

    # No _process_single_waveform needed as Kimi uses paths
    # No _process_single_path needed as core logic uses path directly

    # --- NEW HELPER METHOD ---
    def _get_interactive_caption(self, audio_path: str, desc_prompt: str, channel: str) -> Tuple[str, float]:
        """Helper for interactive modes. Kimi uses paths."""
        start_time = time.time()
        if channel == "left":
            # Need temp file for left channel path
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_l:
                left_wave, _ = audio_utils.get_audio_channels(audio_path)
                if left_wave is None: return "Error loading left channel", 0.0
                sf.write(tmp_l.name, left_wave, audio_utils.TARGET_SAMPLE_RATE)
                # Call core logic with temp path
                caption, _ = self._process_audio_core(tmp_l.name, desc_prompt, "direct-single")
        elif channel == "right":
            # Need temp file for right channel path
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_r:
                _, right_wave = audio_utils.get_audio_channels(audio_path)
                if right_wave is None: return "Error loading right channel", 0.0
                sf.write(tmp_r.name, right_wave, audio_utils.TARGET_SAMPLE_RATE)
                # Call core logic with temp path
                caption, _ = self._process_audio_core(tmp_r.name, desc_prompt, "direct-single")
        else: # channel == "single"
            # Use core logic with original path
            caption, _ = self._process_audio_core(audio_path, desc_prompt, "direct-single")
        processing_time = time.time() - start_time
        return caption, processing_time
    # --- END NEW HELPER ---