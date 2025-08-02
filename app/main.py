import os
import re
import sys
import json
import argparse
import pandas as pd
from app.decoder import parse_dec_file_to_csv

# def load_keywords(file_path):
#     """
#     Load keyword rules from a JSON file.

#     The JSON file must map category names (e.g., 'sql_injection') to strings of space-separated keywords.
#     This function splits each string into a list of keywords.

#     Args:
#         file_path (str): Path to the JSON file containing keyword rules.

#     Returns:
#         dict: A dictionary mapping rule categories to lists of keywords.
#     """
#     with open(file_path, 'r', encoding='utf-8') as f:
#         keywords = json.load(f)

#     for key in keywords:
#         keywords[key] = keywords[key].split()

#     return keywords


# def counting(url, keywords, baseline):
#     """
#     Count keyword occurrences in a URL and assign a label based on threshold.

#     For each rule category, this function counts the number of matching keywords found in the URL.
#     If the highest count exceeds the baseline, the request is labeled as an attack of that type;
#     otherwise, it's labeled as 'benign'.

#     Args:
#         url (str): The URL string to be analyzed.
#         keywords (dict): Dictionary mapping categories to lists of keywords.
#         baseline (int): Minimum keyword matches required to consider as an attack.

#     Returns:
#         pd.Series: A series containing:
#             - Binary label (0 = benign, 1 = attack)
#             - Predicted attack type (or 'benign')
#             - Number of matched keywords
#     """
#     scores = {}
#     for key, words in keywords.items():
#         score = 0
#         for word in words:
#             if word in re.split(r'\W+', url):
#                 score += 1
#         scores[key] = score
        
#     max_score = max(scores, key=scores.get)
#     if scores[max_score] < baseline:
#         return pd.Series([0, 'benign', scores[max_score]])
#     else:
#         return pd.Series([1, max_score, scores[max_score]])


# def inspect(in_path, rules_path, out_path, baseline):
#     """
#     Analyze decoded NGINX logs and label them using keyword-based heuristics.

#     Applies rule-based classification to URLs in the decoded CSV file and writes
#     the result with predictions to a new CSV file.

#     Args:
#         in_path (str): Path to the decoded log CSV file.
#         rules_path (str): Path to the keyword rule JSON file.
#         out_path (str): Path to the output labeled CSV file.
#         baseline (int): Minimum keyword occurrence threshold for attack classification.
#     """
#     df = pd.read_csv(in_path, parse_dates=['time'], encoding='utf-8')
#     print(f"✅ Loaded {len(df)} rows from {in_path}")

#     rules = load_keywords(rules_path)
#     print(f"✅ Loaded {len(rules)} rules from {rules_path}")

#     df[['pred_label', 'pred_type', 'score']] = df['url'].apply(lambda url: counting(url, rules, baseline))
#     df.to_csv(out_path, index=False)    
#     print(f"✅ Log file successfully labeled to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NGINX log labeler.")
    parser.add_argument("in_file", help="NGINX log file")
    # parser.add_argument("rules_file", help="The rule file")
    parser.add_argument("out_file", help="The labeled NGINX csv file")
    parser.add_argument("--baseline", type=int, default=3, help="Minimal word occurrence to be classfied as an attack.")

    # Parse the arguments
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"❌ File not found: '{args.in_file}'")
        sys.exit(1)

    # if not os.path.exists(args.rules_file):
    #     print(f"❌ Rule file not found: '{args.rules_file}'")
    #     sys.exit(1)
    
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        print(f"❌ Output directory is not found: '{out_dir}'")
        sys.exit(1)
    
    parse_dec_file_to_csv(args.in_file, args.out_file)
    # inspect(args.out_file, args.rules_file, args.out_file, args.baseline)