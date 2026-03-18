# evaluator_base.py
import os
import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional

import audio_utils
import prompt_templates
import config 

class BaseEvaluator(ABC):
    """
    Abstract Base Class for all model evaluators.
    Defines the common interface for initialization and evaluation.
    """
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
        print(f"Initializing evaluator for: {self.model_id}")

    @abstractmethod
    def process_audio(
        self,
        audio_path: str,
        prompt_template: str, # Changed: Always pass the template string
        audio_mode: str,
        # Captions passed here are only for captioning-* modes' PRE-FORMATTED text
        # Interactive modes generate captions INTERNALLY using the template
        caption: Optional[str] = None, # Formatted: "Caption: ..." or None
        left_caption: Optional[str] = None, # Formatted: "Left Caption: ..." or None
        right_caption: Optional[str] = None # Formatted: "Right Caption: ..." or None
    ) -> Tuple[str, float]:
        """
        Process a single audio file with the model.
        Subclasses handle interactive caption generation and final prompt formatting.
        Receives the base prompt template string.
        """
        pass # Subclass implements actual logic

    def parse_response(self, response: str, correct_choice: str) -> Dict:
        """ Parses model response to extract choice and check if it's correct. """
        # ... (previous correct parse_response logic remains unchanged) ...
        choice_patterns = [
            r'[Cc]hoice\s*[:\-]?\s*([A-Za-z]+)',
            r'[Aa]nswer\s*[:\-]?\s*([A-Za-z]+)',
            r'\b(Left|Right|Front|Back|Up|Down)\b',
        ]
        choice = "Unknown"
        reasoning = response.strip()
        is_correct = False
        for pattern in choice_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_choice = match.group(1) if match.groups() else match.group(0)
                choice = extracted_choice.title()
                reasoning_start = match.end()
                reasoning = response[reasoning_start:].strip(" :-.,\n")
                if not reasoning: reasoning = response.strip()
                is_correct = choice.lower() == correct_choice.lower()
                break
        if choice == "Unknown":
            if re.search(r'\b' + re.escape(correct_choice) + r'\b', response, re.IGNORECASE):
                choice = correct_choice.title()
                is_correct = True
        return {
            "choice": choice, "correct_choice": correct_choice, "is_correct": is_correct,
            "reasoning": reasoning, "full_response": response
        }


    def evaluate_directory(
        self,
        base_dir: str,
        audio_mode: str,
        iterations: int = 1,
        output_dir: str = "results"
    ) -> Dict:
        """
        Evaluate all audio files in the directory structure.
        Handles caption fetching for captioning-* modes.
        Passes template string to process_audio for interactive-* and direct-* modes.
        """
        model_output_dir = os.path.join(output_dir, self.model_id)
        os.makedirs(model_output_dir, exist_ok=True)

        all_results = {}
        summary_results = {}

        for sound_type in ["SameSound", "DifferentSound"]:
            # ... (folder traversal logic remains the same) ...
             sound_path = os.path.join(base_dir, sound_type)
             if not os.path.exists(sound_path): continue
             for dimension in ["2D", "3D"]:
                  dim_path = os.path.join(sound_path, dimension)
                  if not os.path.exists(dim_path): continue
                  for difficulty in ["Easy", "Mid", "Hard"]:
                       task_path = os.path.join(dim_path, difficulty)
                       if not os.path.exists(task_path): continue

                       task_id = f"{sound_type}_{dimension}_{difficulty}"
                       prompt_template_str = prompt_templates.get_prompt_template(task_id, audio_mode)
                       audio_files = [f for f in os.listdir(task_path) if f.endswith(".wav") and not f.startswith(".")]
                       if not audio_files: continue

                       all_results[task_id] = {}
                       summary_results[task_id] = {"total": 0, "correct": 0, "accuracy": 0.0}

                       for i in range(iterations):
                            iteration_key = f"iteration_{i+1}"
                            all_results[task_id][iteration_key] = {}
                            random.shuffle(audio_files)

                            for audio_file in audio_files:
                                audio_path = os.path.join(task_path, audio_file)
                                correct_choice = os.path.splitext(audio_file)[0]

                                print(f"--- Processing {audio_file} with {self.model_id} (iter {i+1}, mode: {audio_mode}) ---")

                                
                                prompt_to_pass = prompt_template_str # Default: pass template

                               



                                response, p_time, final_prompt = self.process_audio(
                                    audio_path,
                                    prompt_to_pass, 
                                    audio_mode,
                                    
                                )

                                # --- Parse and Store ---
                                parsed_response = self.parse_response(response, correct_choice)
                                summary_results[task_id]["total"] += 1
                                if parsed_response["is_correct"]:
                                    summary_results[task_id]["correct"] += 1

                                # Log the final prompt used if possible (subclass responsibility)
                                # For now, log what was passed + parsed response
                                all_results[task_id][iteration_key][audio_file] = {
                                    "prompt_passed_to_evaluator": final_prompt, # What base class sent
                                    # "final_prompt_used": final_prompt_from_subclass, # Ideal
                                    "processing_time": p_time,
                                    **parsed_response
                                }

                            # ... (save iteration results logic remains the same) ...
                            iter_output_file = os.path.join(model_output_dir, f"{task_id}_{iteration_key}_{audio_mode}.json")
                            with open(iter_output_file, "w") as f: json.dump(all_results[task_id][iteration_key], f, indent=2)
                            print(f"Saved results to {iter_output_file}")

                       if summary_results[task_id]["total"] > 0:
                            summary_results[task_id]["accuracy"] = (summary_results[task_id]["correct"] / summary_results[task_id]["total"])

        return {"detailed_results": all_results, "summary_results": summary_results}