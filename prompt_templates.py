# prompt_templates.py


# --- Main Evaluation Task Prompts ---

PROMPT_TEMPLATES = {
    # === SameSound 2D ===
    "SameSound_2D_Easy": "Task: Identify the direction of the audio.\nOptions: Left, Right\n Provide your choice: ",
    "SameSound_2D_Mid": "Task: Identify the direction of the audio.\nOptions: Left, Right, Up, Down\n Provide your choice: ",
    
    # === SameSound 3D ===
    "SameSound_3D_Easy": "Task: Identify the direction of the audio.\nOptions: Left, Right, Up, Down, Forwards, backwards\n Provide your choice: ",
    "SameSound_3D_Hard": "Task: Identify the direction of the audio.\nOptions: Right-Left, Up-Down, Forward-Backwards\n Provide your choice:",
}


def get_prompt_template(task_id: str, audio_mode: str) -> str:
    """
    Gets the correct prompt template based on task and audio mode.
    Falls back to the single-mode prompt if a double-mode prompt isn't defined.
    """
    # Use an empty caption by default so .format() doesn't fail
    base_prompt = """
    File: {audio_name}
    {caption}
    {left_caption}
    {right_caption}
    Please analyze the audio and provide your answer.
    """
    
    key_to_try = task_id
    if audio_mode.endswith("double"):
        double_key = f"{task_id}_Double"
        if double_key in PROMPT_TEMPLATES:
            key_to_try = double_key
        elif task_id in PROMPT_TEMPLATES:
            key_to_try = task_id
    elif task_id in PROMPT_TEMPLATES:
        key_to_try = task_id
    
    if key_to_try in PROMPT_TEMPLATES:
        base_prompt = PROMPT_TEMPLATES[key_to_try]
    else:
        print(f"Warning: No specific prompt for {task_id}. Using generic prompt.")

    # Pre-fill caption placeholders to avoid errors on .format()
    # if they aren't provided later.
    final_prompt = base_prompt.format(
        caption="{caption}", 
        left_caption="{left_caption}", 
        right_caption="{right_caption}",
        audio_name="{audio_name}"
    )
    # This logic now ensures that if a caption is not provided,
    # the placeholder (and the heading) disappears entirely.
    #final_prompt = final_prompt.replace("{caption}", "").replace("{left_caption}", "").replace("{right_caption}", "").replace("{audio_name}", "")

    return final_prompt
