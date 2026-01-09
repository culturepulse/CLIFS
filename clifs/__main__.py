#!/usr/bin/env python
"""CLI for CLIFS."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import nltk
import getpass
from openai import OpenAI

from clifs.config import CLIFSConfig
from clifs.models import ModelLoader
from clifs.predictors import (
    ClassificationPredictor,
    RegressionPredictor,
    EnsemblePredictor
)


def main():
    parser = argparse.ArgumentParser(
        description="CLIFS: Cognitive Linguistic Identity Fusion Score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic classification
  clifs --input data.csv --output results.csv

  # Regression mode
  clifs --input data.csv --output results.csv --regression

  # Ensemble mode (requires API keys)
  clifs --input data.csv --output results.csv --ensemble

  # With known groups
  clifs --input data.csv --output results.csv --groups religion church usa
        """
    )

    parser.add_argument('--input', '-i', type=Path, required=True,
                       help='Input CSV file with text column')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='Output CSV file path for results')
    parser.add_argument('--groups', '-g', nargs='+', default=[],
                       help='Known target groups for analysis')
    parser.add_argument('--regression', '-r', action='store_true',
                       help='Use regression mode (continuous scores)')
    parser.add_argument('--ensemble', '-e', action='store_true',
                       help='Use ensemble mode (requires API keys)')
    parser.add_argument('--text-column', default='text',
                       help='Name of text column in CSV')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--version', action='version', version='CLIFS 0.1.0')

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load configuration
        config = CLIFSConfig.from_env()
        config.runtime.force_cpu = args.cpu
        config.runtime.setup_environment()
        config.paths.ensure_directories()

        # Load data
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)

        if args.text_column not in df.columns:
            print(f"Error: Column '{args.text_column}' not found", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
            sys.exit(1)

        if args.text_column != 'text':
            df = df.rename(columns={args.text_column: 'text'})

        # Download NLTK data
        nltk.download('punkt', quiet=True)

        # Load models
        print("Loading models...")
        loader = ModelLoader(config)
        models = loader.load_base_models()

        # Create predictor based on mode
        known_groups = tuple(args.groups)

        if args.ensemble:
            print("Setting up ensemble mode...")
            models = loader.add_ensemble(models)

            # Get API keys
            print("\nAPI keys required for ensemble mode:")
            ds_key = getpass.getpass("  DeepSeek API key: ")
            oai_key = getpass.getpass("  OpenAI API key: ")

            ds_client = OpenAI(base_url="https://api.deepseek.com", api_key=ds_key)
            oai_client = OpenAI(api_key=oai_key)

            predictor = EnsemblePredictor(models, known_groups, oai_client, ds_client)
            mode = "ensemble"

        elif args.regression:
            print("Setting up regression mode...")
            models = loader.add_regression(models)
            predictor = RegressionPredictor(models, known_groups)
            mode = "regression"

        else:
            print("Setting up classification mode...")
            predictor = ClassificationPredictor(models, known_groups)
            mode = "classification"

        # Run predictions
        print(f"\nProcessing {len(df)} texts ({mode} mode)...")
        if known_groups:
            print(f"Known groups: {', '.join(known_groups)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            result = predictor.predict_single(row['text'])
            for key, value in result.items():
                df.at[idx, key] = value

            # Checkpoint every 10
            if (idx + 1) % 10 == 0:
                df.to_csv(args.output, index=False)

        # Final save
        df.to_csv(args.output, index=False)
        print(f"\n✓ Success! Results saved to: {args.output}")
        print(f"  Processed {len(df)} texts")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()