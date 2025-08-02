# **Web Server Log URL Clustering with BERT and KMeans**

This application processes **web server log files**, extracts unique URLs, generates **BERT-based embeddings**, and clusters them using **KMeans**.
The tool is designed to help **identify patterns, group similar URLs, and detect anomalous or suspicious paths** in large-scale server logs.

---

## 🧠 How It Works

1. **Extract unique URLs** from the log file.
2. **Normalize URLs:** Replace all numeric IDs with `<NUM>`.
3. **Tokenize:** Break URLs into meaningful tokens.
4. **Embed:** Generate dense vector representations using **BERT embeddings**.
5. **Cluster:** Group URLs with **KMeans**.
6. **Output:** Save results in CSV and TXT formats.

---

## 🚀 Features

* **Log Parsing:** Reads decoded log files and extracts unique URLs.
* **Preprocessing:** Masks numeric sequences (e.g., IDs, timestamps) with `<NUM>` to generalize URL patterns.
* **Tokenization:** Splits URLs into tokens based on common delimiters (paths, query parameters, etc.).
* **Embeddings:** Uses a pre-trained **BERT model** (`bert-base-uncased`) to generate vector representations of URLs.
* **Clustering:** Applies **KMeans** to group URLs with similar structure/content.
* **Output:** Saves results as:

  * A `.csv` file with URLs and their assigned cluster labels.
  * A `.txt` file listing clustered URLs for easy review.

---

## 📂 Project Structure

```
mlwl/
│
├── app/
│   ├── main.py        # Main CLI script for clustering URLs
│   ├── decoder.py     # Utility to parse decoded log files into DataFrame
│
├── inputs/            # Example input log files
├── outputs/           # Output directory for clustering results
└── README.md          # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mlwl.git
   cd mlwl
   ```

2. **Create and activate a Conda environment (Miniconda/Anaconda):**

   ```bash
   conda create -n mlwl_env python=3.11 -y
   conda activate mlwl_env
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```


---

## ▶️ Usage

Run the script via command line:

```bash
python app/main.py <input_log_file> <output_csv_file> -n <num_clusters>
```

### Example:

```bash
python app/main.py inputs/sample.log outputs/clusters.csv -n 8
```

This will:

* Read `inputs/sample.log`
* Cluster unique URLs into **8 groups**
* Save:

  * `outputs/clusters.csv` → URLs with cluster labels
  * `outputs/clusters.csv.txt` → Human-readable clustered URLs

---

## 🐳 Running with Docker

You can use Docker to containerize and run this application without installing dependencies manually.

### **1️⃣ Build the Docker Image**

From the project root directory:

```bash
docker build -t mlwl-app .
```

---

### **2️⃣ Run the Application**

To process a log file and generate clustered results:

```bash
docker run --rm -v $(pwd)/inputs:/app/inputs -v $(pwd)/outputs:/app/outputs mlwl-app python app/main.py inputs/sample.log outputs/sample.csv -n 8
```

* `-v $(pwd)/inputs:/app/inputs` → Mounts the input logs folder.
* `-v $(pwd)/outputs:/app/outputs` → Mounts the output folder.
* `-n 8` → Number of clusters to generate.

---

### **3️⃣ Verify the Output**

After completion, the results will be available in your local `outputs/` directory:

* `clusters.csv` → CSV file with URLs and their assigned cluster labels.
* `clusters.csv.txt` → Text file listing clustered URLs for quick inspection.