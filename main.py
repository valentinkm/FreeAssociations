import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src import settings, data_utils, llm_generation, generalizability, profile_utils, experiment_runners, evaluation
except ImportError as e:
    print(f"Error: Could not import necessary modules from 'src'. Details: {e}")
    print("Please ensure you have run 'pip install google-generativeai openai pandas seaborn matplotlib statsmodels tqdm'")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="A command-line tool for the FreeAssociations project.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output, including API call details.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: generalize
    parser_generalize = subparsers.add_parser("generalize", help="Run a full generalization analysis pipeline.", parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_generalize.add_argument("--model", type=str, default=settings.CFG['model'], help="Model for generation (e.g., gpt-4o, gemini-1.5-flash-latest, llama3).")
    parser_generalize.add_argument("--type", type=str, required=True, choices=["spp", "3tt"], help="The generalization target.")
    parser_generalize.add_argument("--nsets", type=int, default=25, help="Association sets per word, determines lexicon file.")
    parser_generalize.add_argument("--ncues", type=int, default=50, help="Number of cues/triplets to sample for the analysis.")
    parser_generalize.add_argument("--prompt", type=str, default="participant_default_question", help="Prompt template for generation.")
    parser_generalize.add_argument("--force-generate", action="store_true", help="Force regeneration of all required words.")

    # Command: generate
    parser_generate = subparsers.add_parser("generate", help="Run a standalone LLM data generation task.", parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_generate.add_argument("--model", type=str, default=settings.CFG['model'], help="Model for generation.")
    parser_generate.add_argument("--type", type=str, required=True, choices=["prompt-sweep", "yoked"], help="The type of generation.")
    parser_generate.add_argument("--nsets", type=int, default=5, help="Number of association sets to generate per cue.")
    parser_generate.add_argument("--ncues", type=int, default=10, help="Number of cues to sample for the experiment.")

    # Command: evaluate
    parser_evaluate = subparsers.add_parser("evaluate", help="Evaluate the results of a generation experiment.", parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_evaluate.add_argument("--type", type=str, required=True, choices=["prompt-sweep", "yoked-steering", "model-comparison", "model-alignment"], help="The experiment type to evaluate.")
    parser_evaluate.add_argument("--task", type=str, default="spp", choices=["spp", "3tt"], help="For model-comparison, which task to compare performance on.")
    parser_evaluate.add_argument("--nsets", type=int, default=25, help="For model-alignment, the number of sets per word for the lexicons to compare.")
    parser_evaluate.add_argument("--ncues", type=int, default=100, help="For model-alignment, the number of cues to ensure exist in each lexicon.")
    parser_evaluate.add_argument("--models", nargs='+', default=["gpt-4o", "llama3"], help="List of models to include in the comparison.")

    # Command: extract-profiles
    parser_profiles = subparsers.add_parser("extract-profiles", help="Extract unique demographic profiles from SWOW data.", parents=[parent_parser])
    parser_profiles.add_argument("--top-countries", type=int, default=20, help="Limit to N most frequent countries (0 for all).")

    args = parser.parse_args()
    
    settings.CFG['verbose'] = args.verbose if hasattr(args, 'verbose') else False
    settings.initialize_project_paths()

    if args.command == "generalize":
        handle_generalize(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "evaluate":
        handle_evaluate(args)
    elif args.command == "extract-profiles":
        handle_extract_profiles(args)

def handle_generalize(args):
    print(f"--- Running Generalization Analysis ---")
    print(f"Model: {args.model} | Target: {args.type.upper()} | Prompt: {args.prompt} | Sets: {args.nsets} | Cues: {args.ncues}")
    lexicon_path = settings.get_lexicon_path(model_name=args.model, nsets=args.nsets)
    if args.type == 'spp':
        data, vocab = data_utils.get_spp_data_for_analysis(lexicon_path, args.ncues)
        vocab_needed_for_force = set(data['prime'].str.lower()).union(set(data['target'].str.lower())) if data is not None else set()
    else:
        data, vocab = data_utils.get_3tt_data_for_analysis(lexicon_path, args.ncues)
        vocab_needed_for_force = set(data['cue'].str.lower()).union(set(data['choiceA'].str.lower())).union(set(data['choiceB'].str.lower())) if data is not None else set()
    
    if data is None: return

    if vocab or args.force_generate:
        vocab_to_run = vocab_needed_for_force if args.force_generate else vocab
        llm_generation.generate_lexicon_data(vocab_to_run, args.model, lexicon_path, args.nsets, args.prompt)
    
    if args.type == 'spp':
        generalizability.analyze_holistic_similarity(data, lexicon_path)
    else:
        generalizability.analyze_3tt_similarity(data, lexicon_path)

def handle_generate(args):
    if args.type == "prompt-sweep":
        experiment_runners.run_sweep(model_name=args.model, nsets=args.nsets, ncues=args.ncues)
    elif args.type == "yoked":
        experiment_runners.run_yoked_generation(model_name=args.model, ncues=args.ncues, nsets=100)

def handle_evaluate(args):
    print(f"--- Running Evaluation: {args.type} ---")
    if args.type == "prompt-sweep":
        evaluation.evaluate_prompt_sweep()
    elif args.type == "yoked-steering":
        evaluation.evaluate_yoked_steering()
    elif args.type == "model-comparison":
        generalizability.compare_models_on_task(task=args.task)
    elif args.type == "model-alignment":
        generalizability.compare_model_alignment(models=args.models, nsets=args.nsets, ncues=args.ncues)

def handle_extract_profiles(args):
    profile_utils.extract_all_profiles(top_countries=args.top_countries if args.top_countries > 0 else None)

if __name__ == "__main__":
    main()
