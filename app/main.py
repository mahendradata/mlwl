import argparse
import pandas as pd
import os

def parse_log_line(line):
    # Simple parser: splits line by spaces
    parts = line.strip().split()
    return parts

def main():
    parser = argparse.ArgumentParser(description="Parse web server log file to CSV")
    parser.add_argument('input_file', help='Path to the input log file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        exit(1)

    parsed_data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                parsed_data.append(parse_log_line(line))

    # Create DataFrame, dynamically handle column count
    max_cols = max(len(row) for row in parsed_data)
    col_names = [f'col{i+1}' for i in range(max_cols)]
    df = pd.DataFrame(parsed_data, columns=col_names)

    # Write DataFrame to output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Parsed log saved to {args.output_file}")

if __name__ == "__main__":
    main()
