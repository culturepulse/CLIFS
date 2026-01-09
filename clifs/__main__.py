#!/usr/bin/env python
"""
Command-line interface for CLIFS (Cognitive Linguistic Identity Fusion Score).
"""
import argparse
import sys
import pandas as pd
from pathlib import Path


def main():
    """Main entry point for the CLIFS CLI."""
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

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input CSV file with "text" column'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path for results'
    )

    parser.add_argument(
        '--groups', '-g',
        nargs='+',
        default=[],
        help='Known target groups for analysis (space-separated)'
    )

    parser.add_argument(
        '--regression', '-r',
        action='store_true',
        help='Use regression mode (continuous scores) instead of classification'
    )

    parser.add_argument(
        '--ensemble', '-e',
        action='store_true',
        help='Use ensemble mode (requires OpenAI and DeepSeek API keys)'
    )

    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of the text column in input CSV (default: "text")'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='CLIFS 0.1.0'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error: Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate text column
    if args.text_column not in df.columns:
        print(
            f"Error: Column '{args.text_column}' not found in input CSV. "
            f"Available columns: {', '.join(df.columns)}",
            file=sys.stderr
        )
        sys.exit(1)

    # Rename column to 'text' if different
    if args.text_column != 'text':
        df = df.rename(columns={args.text_column: 'text'})

    # Import and run CLIFS
    try:
        from clifs import clifs

        print(f"Processing {len(df)} texts...")
        print(f"Mode: {'Ensemble' if args.ensemble else 'Regression' if args.regression else 'Classification'}")
        if args.groups:
            print(f"Known groups: {', '.join(args.groups)}")

        results = clifs.clifs(
            df=df,
            known_groups=args.groups,
            ensemble=args.ensemble,
            regression=args.regression,
            save_path=args.output
        )

        print(f"\nSuccess! Results saved to: {args.output}")
        print(f"Processed {len(results)} texts")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()