import pandas as pd
import re

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

def parse_log(input_file):
    parsed_data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                parsed_line = parse_log_line(line)
                if parsed_line:
                    parsed_data.append(parsed_line)

    if not parsed_data:
        print("No valid log entries found.")
        exit(0)

    df = pd.DataFrame(parsed_data)
    print(f"[INFO] parse_log({input_file}) done: {df.shape[0]} lines.")
    return df
    