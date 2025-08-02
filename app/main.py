import os
import re
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse, unquote
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from app.decoder import parse_dec_file_to_dataframe
from pprint import pprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Step 1: Get Unique URLs
def url_for_training(df):
    result = df['url'].unique()
    result = [re.sub(r'\d+', '<NUM>', url) for url in result]
    result = set(result)   
    return list(result)


# Step 2: Clean and tokenize URLs
def tokenize_url(url):
    parsed = urlparse(url)
    path = unquote(parsed.path)
    query = unquote(parsed.query)

    # Extended the regex to include (), [], <>
    delimiters = r"[\/\-\_\=\&\?\.\+\(\)\[\]\<\>\{\}]"
    path_tokens = re.split(delimiters, path.strip("/"))
    query_tokens = re.split(delimiters, query)

    tokens = [tok for tok in path_tokens + query_tokens if tok]
    return tokens


# Step 3: Get BERT embeddings
def get_url_embedding(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # back to CPU


def clustering(df, out_path, n_clusters):
    train_url_list = url_for_training(df)
    pprint(train_url_list)
    print(f"✅ train_url_list {len(train_url_list)}")

    tokenized_urls = [" ".join(tokenize_url(url)) for url in train_url_list]
    pprint(tokenized_urls)
    print(f"✅ tokenized_urls {len(tokenized_urls)}")

    embeddings = np.array([get_url_embedding(url) for url in tokenized_urls])
    # Step 4: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Step 5: Print Clustered URLs
    clustered_urls = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_urls[label].append(train_url_list[idx])

    with open("clustered_urls.txt", "w") as f:
        for cluster, urls in clustered_urls.items():
            f.write(f"\nCluster {cluster}:\n")
            for url in urls:
                f.write(f"  {url}\n")

    # Step 6: Save clustered URLs to CSV
    df_label = pd.DataFrame({
        "masked": train_url_list,
        "cluster": labels
    })
    df_label = df_label.sort_values(by='cluster')
    df_label.to_csv(out_path, index=False, encoding="utf-8")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NGINX log labeler.")
    parser.add_argument("in_file", help="NGINX log file")
    parser.add_argument("out_file", help="The labeled NGINX csv file")
    parser.add_argument("-n", type=int, default=10, help="Number of cluster.")

    # Parse the arguments
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"❌ File not found: '{args.in_file}'")
        sys.exit(1)

    out_dir = os.path.dirname(args.out_file)
    if out_dir and not os.path.exists(out_dir):
        print(f"❌ Output directory is not found: '{out_dir}'")
        sys.exit(1)
    
    df = parse_dec_file_to_dataframe(args.in_file)
    print(f"✅ Loaded {len(df)} rows from {args.in_file}")

    clustering(df, args.out_file, args.n)