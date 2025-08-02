import argparse
import pandas as pd
import os
import re
from app.parser import parse_log

log_pattern = re.compile(
    r'(?P<ip>\S+) - - \[(?P<time>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<url>\S+) (?P<protocol>[^"]+)" '
    r'(?P<status>\d+) (?P<size>\d+) '
    r'"(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)" "(?P<extra>[^"]*)"'
)

def parse_log_line(line):
    match = log_pattern.match(line)
    if match:
        return match.groupdict()
    return None

def main():
    parser = argparse.ArgumentParser(description="Parse web server log file to CSV")
    parser.add_argument('input_file', help='Path to the input log file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        exit(1)

    df = parse_log(args.input_file)

    # Ensure output directory exists (create even if empty string)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output_file, index=False)
    print(f"[INFO] Parsed log saved to {args.output_file}")

if __name__ == "__main__":
    main()
