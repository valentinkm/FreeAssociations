import argparse
import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import the new, refactored modules directly.
try:
    from src import settings
    from src import data_utils
    from src import llm_generation
    from src import analysis
    from src import profile_utils
    from src import experiment_runners
except ImportError as e:
    print("Error: Could not import necessary modules from 'src'.")
    print(f"Details: {e}")
    print("\nPlease ensure you have created all the required files in the 'src' directory:")
    print("  - src/settings.py")
    print("  - src/data_utils.py")
    print("  - src/llm_generation.py")
    print("  - src/analysis.py")
    print("  - src/profile_utils.py")
    print("  - src/experiment_runners.py")
    sys.exit(1)


def main():
    """
    Main entry point for the FreeAssociations project command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="A command-line tool to run analyses and data generation for the FreeAssociations project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command: analyze
    parser_analyze = subparsers.add_parser("analyze", help="Run a data analysis pipeline.")
    parser_analyze.add_argument("--type", type=str, required=True, choices=["llm-holistic", "human-baseline"], help="The type of analysis to perform.")
    parser_analyze.add_argument("--force-generate", action="store_true", help="For 'llm-holistic', force regeneration of the lexicon.")

    # Command: generate
    parser_generate = subparsers.add_parser("generate", help="Run an LLM data generation task.")
    parser_generate.add_argument("--type", type=str, required=True, choices=["lexicon", "yoked", "sweep"], help="The type of generation to perform.")
    parser_generate.add_argument("--nsets", type=int, default=5, help="For 'sweep', the number of association triples to generate per cue.")
    # Command: extract-profiles
    parser_profiles = subparsers.add_parser("extract-profiles", help="Extract and save unique demographic profiles from SWOW data.")
    parser_profiles.add_argument("--top-countries", type=int, default=20, help="Number of most frequent countries to include (0 for all).")

    args = parser.parse_args()

    # Initialize paths and settings from the settings module
    settings.initialize_project_paths()
    
    # Execute the requested command
    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "extract-profiles":
        handle_extract_profiles(args)
    else:
        parser.print_help()


def handle_analyze(args):
    """Handler for the 'analyze' command."""
    print(f"--- Running Analysis: {args.type} ---")
    if args.type == "llm-holistic":
        # CORRECTED: Call the function from the data_utils module
        spp_data, vocab_to_generate = data_utils.get_spp_data_for_analysis()
        
        if vocab_to_generate or args.force_generate:
            # If forcing, regenerate for the full vocabulary of the sampled pairs
            if args.force_generate:
                print("Forcing lexicon regeneration for all words in the sampled SPP dataset.")
                full_vocab = set(spp_data['prime'].str.lower()).union(set(spp_data['target'].str.lower()))
                vocab_to_run = full_vocab
            else:
                vocab_to_run = vocab_to_generate
            
            llm_generation.generate_lexicon_data(vocab_to_run)
        else:
             print("✅ Lexicon is already up-to-date.")

        analysis.analyze_holistic_similarity(spp_data)

    elif args.type == "human-baseline":
        analysis.analyze_human_baseline()


def handle_generate(args):
    """Handler for the 'generate' command."""
    print(f"--- Running Generation: {args.type} ---")
    if args.type == "lexicon":
        print("This command is now run implicitly via 'analyze --type=llm-holistic'.")
        print("To force regeneration, use 'analyze --type=llm-holistic --force-generate'")
    elif args.type == "yoked":
        experiment_runners.run_yoked_generation()
    elif args.type == "sweep":
        experiment_runners.run_sweep(nsets=args.nsets, no_demo=args.no_demo)


def handle_extract_profiles(args):
    """Handler for the 'extract-profiles' command."""
    print("--- Extracting Demographic Profiles ---")
    profile_utils.extract_all_profiles(top_countries=args.top_countries if args.top_countries > 0 else None)


if __name__ == "__main__":
    main()
