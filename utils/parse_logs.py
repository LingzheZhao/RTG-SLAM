#!/usr/bin/env python3

import re
import pandas as pd
import argparse
from pathlib import Path
import sys

def parse_slam_output(text: str) -> pd.DataFrame:
    pattern = r"run (\S+)\nSystem FPS: ([\d.]+)\nATE RMSE: ([\d.]+)\nPSNR: ([\d.]+)\nSSIM: ([\d.]+)\nLPIPS: ([\d.]+)"
    matches = re.findall(pattern, text)

    columns = ["Sequence", "FPS", "ATE_RMSE", "PSNR", "SSIM", "LPIPS"]
    df = pd.DataFrame(matches, columns=columns)

    # Convert numeric columns
    for col in columns[1:]:
        df[col] = df[col].astype(float)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Parse SLAM system performance output into a table.\n"
        "Example: python parse_slam_output.py slam_output.txt -o results.csv"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input text file containing SLAM output."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional path to save the parsed CSV file."
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    with args.input_file.open("r", encoding="utf-8") as f:
        content = f.read()

    df = parse_slam_output(content)
    print(df)

    output_path: Path = args.output
    if output_path:
        if output_path.exists():
            print("[WARN] file exists")
            return 1
        df.to_csv(args.output, index=False, sep='\t')
        print(f"\nSaved parsed results to '{args.output}'")

if __name__ == "__main__":
    main()
