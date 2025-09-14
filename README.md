# Jenosize Pipeline & FastAPI Service

This repository provides tools to **scrape, preprocess, fine-tune, and deploy** content for LLMs, including a containerized FastAPI service for inference.

---

## 1. Prerequisites

Before starting, make sure you have:

* **Python 3.11+**
* Optional: a code editor (recommended: `VSCode`)
* **Chrome** or another browser supported by Selenium
* **Docker & Docker Daemon** (for FastAPI deployment)
* **NVIDIA GPUs** and NVIDIA Container Toolkit (for GPU acceleration in Docker)

---

You need to create a virtual environment and install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Data Pipeline & Preprocessing

This pipeline scrapes articles, cleans the content, and outputs a structured JSON ready for fine-tuning.

### Step 1: Build the Data Pipeline

1. Open the Jupyter Notebook (`.ipynb`) and select the kernel you created (e.g., `jenosize_interview/.venv`).
2. Follow the notebook steps to scrape and process articles.

**Output File:**

```
./datasets/jenosize-article-mockup.json
```

### Input Format

Each item in the JSON should have the following keys:

* `topic_category`
* `industry`
* `target_audience`
* `website`
* `seo_keywords`
* `content` (used as labels)

**Example JSON:**

```json
[
  {
    "topic_category": ["AI", "Education"],
    "industry": ["Technology", "Education"],
    "target_audience": ["Teachers", "Students"],
    "website": "https://example.com",
    "seo_keywords": ["AI trends", "future of education"],
    "content": "# Artificial intelligence is transforming how students learn ..."
  }
]
```

**Example JSONL (one conversation per line):**

```json
{
    "topic_category": ["AI", "Education"], 
    "industry": ["Technology", "Education"], 
    "target_audience": ["Teachers", "Students"], 
    "website": "https://example.com", 
    "seo_keywords": ["AI trends", "future of education"], 
    "content": "# Artificial intelligence is transforming how students learn ..."
}
```

---

### Step 2: Prepare Conversations for Fine-Tuning

Run the following command:

```bash
python -m src.data.prepare_conversation \
    --input-path ./datasets/jenosize-article-mockup.json \
    --save-path ./datasets/train/train-conversations.jsonl
```

**Arguments:**

* `--input-path`: Path to input dataset file (`.json` or `.jsonl`).
* `--save-path`: Path to save processed conversations (`.json` or `.jsonl`).

**Output Structure:**

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "<instruction prompt with metadata>"},
    {"from": "gpt", "value": "<original content from dataset>"}
  ]
}
```

---

## 3. Fine-Tuning Instructions

Before we go to the steps,

---

## Model Selection Rationale

After reviewing research papers, public benchmarks, and experimenting with multiple models of different sizes, I shortlisted several candidates: **Google Gemma3**, **OpenAI GPT models**, and **Qwen3-8B**.  

I selected **Qwen3-8B** for fine-tuning because it provides an optimal balance between model quality and computational efficiency.  

- Compared to **Google’s Gemma3**, it is lighter and faster to fine-tune while still producing competitive text generation quality.  
- Unlike **OpenAI GPT models**, Qwen3 is fully open-source, enabling local fine-tuning and GPU-based deployment without vendor lock-in.  

This makes **Qwen3-8B** a practical and cost-effective choice for building the prototype.

---

1. Configure parameters in `scripts/finetune.sh`, including:

* `MODEL_NAME` (LLM model from Hugging Face)
* `GLOBAL_BATCH_SIZE`, `BATCH_PER_DEVICE`, `NUM_DEVICES`
* Checkpoint naming, output directories, and DeepSpeed settings

2. Run fine-tuning:

```bash
bash scripts/finetune.sh
```

**Notes:**

* Use `tmux` or `screen` to run long training sessions in the background.
* Logs are saved in:

```txt
./output/output_logs/history_log_${CHECKPOINT_NAME}.out
```

---

## 4. Deploy with FastAPI

This project includes a containerized FastAPI service optimized for NVIDIA GPUs and Hugging Face integration.

### Step 1: Check GPU and CUDA version

```bash
nvidia-smi
```

Select the appropriate CUDA base image in the `Dockerfile`.

### Step 2: Create `.env` File

```env
HF_HOME=[your-hf-cache-directory]
HUGGING_FACE_HUB_TOKEN=[your-hf-token]
```

### Step 3: Configure GPU Allocation

Edit `docker-compose.yml`:

```yaml
...
deploy:
  resources:
    reservations:
      devices:
      - driver: nvidia
        capabilities: [gpu]
        device_ids: ['0']   # Replace with your GPU IDs
```

### Step 4: Build and Run Container

```bash
sudo docker compose up --build
sudo docker compose up --build -d   # Run in background
```

### Step 5: Access the Service

FastAPI service will be available at:

```
http://localhost:80
```

(Default internal port `12999` is mapped to `80`)

---
