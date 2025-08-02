import os
import re
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse, unquote
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from app.decoder import parse_dec_file_to_dataframe
from pprint import pprint

# Select GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model ONCE globally for efficiency
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL = BertModel.from_pretrained("bert-base-uncased").to(device)
MODEL.eval()

def extract_unique_urls(df):
    """
    Extract unique URLs from the dataframe and mask numeric values with <NUM>.
    """
    result = df['url'].unique()
    result = [re.sub(r'\d+', '<NUM>', url) for url in result]
    return list(set(result))

def split_url_tokens(url):
    """
    Tokenize a URL by splitting on common delimiters (path and query string).
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)
    query = unquote(parsed.query)
    delimiters = r"[\/\-\_\=\&\?\.\+\(\)\[\]\<\>\{\}]"
    tokens = re.split(delimiters, path.strip("/")) + re.split(delimiters, query)
    return [tok for tok in tokens if tok]

def generate_url_embeddings(url_list, batch_size=16):
    """
    Generate BERT embeddings for a list of preprocessed URLs using batched inference.
    """
    embeddings = []
    for i in range(0, len(url_list), batch_size):
        batch = url_list[i:i+batch_size]
        inputs = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            outputs = MODEL(**inputs)
        # Average token embeddings
        batch_emb = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def cluster_urls_from_log(df, out_path, n_clusters):
    """
    Perform clustering on URLs using BERT embeddings and KMeans.
    Saves results in both text and CSV format.
    """
    # Step 1: Extract unique URLs
    unique_urls = extract_unique_urls(df)
    # Print the unique_urls for debuging purpose
    print(f"✅ Output of: unique_urls")
    pprint(unique_urls)

    # Step 2: Tokenize each URL into text
    tokenized_urls = [" ".join(split_url_tokens(url)) for url in unique_urls]
    # Print the tokenized_urls for debuging purpose
    print(f"✅ Output of: tokenized_urls")
    pprint(tokenized_urls)

    # Step 3: Convert URLs to embeddings
    embeddings = generate_url_embeddings(tokenized_urls)
    embeddings = normalize(embeddings)
    # Print the embeddings for debuging purpose
    print(f"✅ Output of: embeddings")
    pprint(embeddings)

    # Step 4: Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    # Print the labels for debuging purpose
    print(f"✅ Output of: labels")
    pprint(labels)

    # Step 5: Group URLs by cluster
    clustered_urls = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_urls[label].append(unique_urls[idx])

    # Step 6: Save clusters to a text file
    with open(f"{out_path}.txt", "w", encoding="utf-8") as f:
        for cluster, urls in clustered_urls.items():
            f.write(f"\nCluster {cluster}:\n")
            for url in urls:
                f.write(f"  {url}\n")

    # Step 7: Save results to CSV
    df_label = pd.DataFrame({"masked": unique_urls, "cluster": labels}).sort_values(by='cluster')
    df_label.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Clustering results saved to: {out_path}")

if __name__ == "__main__":
    """
    CLI for NGINX log clustering tool.

    Example:
        python main.py inputs/sample.log outputs/clusters.csv -n 8
    """
    parser = argparse.ArgumentParser(description="Cluster NGINX log URLs using BERT embeddings.")
    parser.add_argument("in_file", help="NGINX log file")
    parser.add_argument("out_file", help="The labeled NGINX CSV file")
    parser.add_argument("-n", type=int, default=10, help="Number of clusters.")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.in_file):
        print(f"❌ File not found: '{args.in_file}'")
        sys.exit(1)

    # Validate output directory
    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        print(f"❌ Output directory not found: '{out_dir}'")
        sys.exit(1)
    
    # Load and process log file
    df = parse_dec_file_to_dataframe(args.in_file)
    print(f"✅ Loaded {len(df)} rows from {args.in_file}")

    # Run clustering process
    cluster_urls_from_log(df, args.out_file, args.n)
