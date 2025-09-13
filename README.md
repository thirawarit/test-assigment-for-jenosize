# Submission Requirements:
1.  A zipped folder or GitHub repository containing:
    * All code files (model fine-tuning, data pipeline, API deployment).
    * Dataset used for fine-tuning (or a link if sourced externally).
    * README file with setup instructions.
2.  A short report (1-2 pages) explaining your approach, challenges faced, and potential improvements.
3.  A Link to test App prototype as a user.

---

## Prerequisites

*   Required `Python 3.11+`
*   Optional required a code editor (suggest `vscode`)

---

# Preparation conversation

## Input Format

The input dataset must contain the following keys for each item:

* `topic_category`
* `industry`
* `target_audience`
* `website`
* `seo_keywords`
* `content` (It is `labels`)

### Example `.json`

```json
[
  {
    "topic_category": ["AI", "Education"],
    "industry": ["Technology", "Education"],
    "target_audience": ["Teachers", "Students"],
    "website": "https://example.com",
    "seo_keywords": ["AI trends", "future of education"],
    "content": "# Artificial intelligence is transforming how students learn in classrooms ..."
  }
]
```

### Example `.jsonl`

```json
{
    "topic_category": ["AI", "Education"], 
    "industry": ["Technology", "Education"], 
    "target_audience": ["Teachers", "Students"], 
    "website": "https://example.com", 
    "seo_keywords": ["AI trends", "future of education"], 
    "content": "Artificial intelligence is transforming how students learn in classrooms."}
```

## Usage

Run the script from the command line:

```bash
python prepare_conversation.py --input-path <INPUT_FILE> --save-path <OUTPUT_FILE>
```

### Arguments

* `--input-path`
  Path to the input dataset file. Must be `.json` or `.jsonl`.

* `--save-path`
  Path to save the processed conversations. Must be `.json` or `.jsonl`.

---

## Example

Convert a JSON dataset to JSONL conversations:

```bash
python prepare_conversation.py \
    --input-path ./datasets/raw.json \
    --save-path ./datasets/train/train_conversations.jsonl
```

Convert a JSONL dataset to JSON:

```bash
python prepare_conversation.py \
    --input-path ./datasets/raw.jsonl \
    --save-path ./datasets/train/train_conversations.json
```

---

## Output

* If `--save-path` ends with `.jsonl`, output will contain one conversation per line.
* If `--save-path` ends with `.json`, output will be a list of conversations in a single JSON file.

Each conversation follows the structure:

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

# Setup Instructions

1.  Create your virtual environment. Activate the venv and install dependent libraries.
```bash
python3.10 -m venv .venv # Optional to change your venv name (".venv")
source .venv/bin/activate # You must change ".venv" here also, if you changed it previously.
pip install -r requirements.txt
```

**NOTE:** If you need to exit from this venv, you can use:
```bash
deactivate
``` 

---

2. Open the file `scripts/finetune.sh` and configure the parameters as needed.
Default settings in the file are shown below:
```bash
MODEL_NAME="Qwen/Qwen3-8B"  # LLM model you selected from Hugging Face

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=2          # Batch size per GPU
NUM_DEVICES=2               # Number of GPUs to use for fine-tuning
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
CHECKPOINT_NAME='trial-qwen3-sft-jenosize-0_0_0b'   # Name for the fine-tuned model
export PYTHONPATH=src:$PYTHONPATH

mkdir -p ./output/output_logs

deepspeed ./src/train/train_sft.py \
    --deepspeed ./scripts/zero3.json \      # You can choose another DeepSpeed config if needed
    --model_id $MODEL_NAME \
    --data_path datasets/train/*.jsonl \    # Path to the training dataset
    --remove_unused_columns False \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir ./output/${CHECKPOINT_NAME} \
    --num_train_epochs 1 \                  # Number of training epochs
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --learning_rate 1e-5 \                  # Learning rate
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
        2>&1 | tee ./output/output_logs/history_log_${CHECKPOINT_NAME}.out
```

---

3. Run `scripts/finetune.sh`. Execute the following command:
```bash
bash scripts/finetune.sh
```
**NOTE:** When running fine-tuning from the terminal, it is common to use tools like `tmux` or `screen`. These allow you to keep the process running in the background even if your terminal session disconnects. We highly recommend learning to use them, as they are very helpful for long-running training jobs.

---

4. All logs from the fine-tuning process are saved in:

```
./output/output_logs/history_log_${CHECKPOINT_NAME}.out
```

You can view them using any text editor, for example with `vim`:

```bash
vim ./output/output_logs/history_log_${CHECKPOINT_NAME}.out
```

---
