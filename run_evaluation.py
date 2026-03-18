# run_evaluation.py
import argparse
import json
import os
from evaluator_factory import get_evaluator # Still need the factory
from config import MODEL_CONFIGS
import traceback # For better error printing

# Define all possible evaluation modes
ALL_MODES = [
    "direct-single",
]

# MODIFIED: Now accepts the initialized evaluator object
def run_single_mode(evaluator, model_id, data_dir, output_dir, iterations, mode):
    """Runs the evaluation for a single specified mode USING a pre-loaded evaluator."""
    print(f"\n--- Running Mode: {mode} ---")

    # Evaluator is already loaded, proceed to evaluation
    try:
        results = evaluator.evaluate_directory(
            data_dir,
            mode,
            iterations,
            output_dir
        )
    except Exception as e:
        print(f"!!! ERROR during evaluation for mode {mode}: {e}")
        traceback.print_exc() # Print full traceback
        print(f"Skipping mode {mode} due to error.")
        return None # Return None on error

    # Save results for this mode
    model_output_dir = os.path.join(output_dir, model_id) # Use model_id for path
    # Ensure directory exists (might be the first mode run)
    os.makedirs(model_output_dir, exist_ok=True)
    consolidated_output = os.path.join(model_output_dir, f"consolidated_results_{mode}.json")
    summary_output = os.path.join(model_output_dir, f"summary_results_{mode}.json")

    try:
        if results and "summary_results" in results:
            with open(consolidated_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Consolidated results saved to {consolidated_output}")

            with open(summary_output, "w") as f:
                json.dump(results["summary_results"], f, indent=2)
            print(f"Summary results saved to {summary_output}")
            print(f"--- Mode {mode} Summary ---")
            print(json.dumps(results["summary_results"], indent=2))
            return results["summary_results"]
        else:
             print(f"!!! Warning: Evaluation for mode {mode} produced invalid or empty results.")
             return None
    except Exception as e:
        print(f"!!! ERROR saving results for mode {mode}: {e}")
        traceback.print_exc() # Print traceback for saving errors too
        return None


def main():
    parser = argparse.ArgumentParser(description="Unified Audio Model Evaluation Framework")

    parser.add_argument(
        "--model", type=str, required=True, choices=MODEL_CONFIGS.keys(),
        help="Name of the model to evaluate."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the root directory containing the audio data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory path (default: results)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=ALL_MODES + ["all"],
        help="Evaluation mode(s) to run. 'all' runs every mode sequentially."
    )
    parser.add_argument(
        "--stepaudio_base_path", type=str, default=None,
        help="Required for '--model step-audio'. Path to the directory containing the downloaded 'Step-Audio-Tokenizer' and 'Step-Audio-Chat' folders."
    )

    args = parser.parse_args()

    # --- Initial Setup and Validation ---
    if args.model == "step-audio":
        if args.stepaudio_base_path is None:
            parser.error("--stepaudio_base_path is required when --model is 'step-audio'")
        if not os.path.isdir(args.stepaudio_base_path):
             parser.error(f"--stepaudio_base_path '{args.stepaudio_base_path}' is not a valid directory.")

    print(f"--- Starting Evaluation ---")
    print(f"Model: {args.model}")
    print(f"Mode(s): {args.mode}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Iterations: {args.iterations}")
    if args.model == "step-audio":
        print(f"StepAudio Base Path: {args.stepaudio_base_path}")
    print(f"---------------------------")

    # --- Load Model ONCE ---
    evaluator_kwargs = {}
    if args.model == "step-audio":
         evaluator_kwargs['stepaudio_base_path'] = args.stepaudio_base_path

    print("Initializing evaluator and loading model...")
    try:
        # Get the evaluator instance (this loads the model)
        evaluator = get_evaluator(args.model, **evaluator_kwargs)
        if evaluator is None:
            # Factory already printed error, just exit
            print(f"Failed to initialize evaluator for {args.model}. Exiting.")
            return # Exit if model loading failed
    except Exception as e:
        print(f"!!! FATAL ERROR during evaluator initialization for {args.model}: {e}")
        traceback.print_exc()
        print("Exiting.")
        return # Exit if model loading failed spectacularly

    print("Model loaded successfully.")
    # --- End Load Model ONCE ---


    overall_summary = {}

    if args.mode == "all":
        modes_to_run = ALL_MODES
        print(f"Running all modes: {', '.join(modes_to_run)}")
        for current_mode in modes_to_run:
            # Pass the SAME evaluator object to each run
            summary = run_single_mode(
                evaluator, args.model, args.data_dir, args.output_dir, args.iterations, current_mode
            )
            if summary is not None:
                overall_summary[current_mode] = summary
            print(f"---------------------------") # Separator

    else:
        # Run only the specified mode with the pre-loaded evaluator
        summary = run_single_mode(
            evaluator, args.model, args.data_dir, args.output_dir, args.iterations, args.mode
        )
        if summary is not None:
            overall_summary[args.mode] = summary

    print(f"\n--- Evaluation Completed ---")
    if args.mode == 'all' and overall_summary:
        print("\n--- Overall Summary Across Modes ---")
        print(json.dumps(overall_summary, indent=2))
        print(f"---------------------------")
    elif not overall_summary:
         print("Evaluation finished with errors or no modes run successfully.")
    else: # Print summary even if only one mode was run
         print("\n--- Final Summary ---")
         print(json.dumps(overall_summary, indent=2))
         print(f"---------------------------")


if __name__ == "__main__":
    main()
