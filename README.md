# Beaker Volume Recognition (Qwen2.5-VL + Unsloth)

This project fine-tunes a Vision–Language Model (Qwen2.5-VL) to estimate liquid volume in beakers.

**Goal output format (strict):**
<number> mL

yaml
Copy code
Examples: `32.5 mL`, `100 mL`, `7 mL`

---

## 1) Setup (JarvisLab / Linux)

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_FOLDER>
pip install -r requirements.txt
Run:

bash
Copy code
python app.py --share
Optional:

bash
Copy code
python app.py --port 7860 --share
2) Using the App
Step A — Load the model
Open the Model tab:

Base model: unsloth/Qwen2.5-VL-3B-Instruct

Keep 4-bit enabled (recommended)

Click Load Model

Click Prepare LoRA for Training

Step B — Load your dataset
Open the Dataset tab:

Set your dataset repo, e.g.:

yusufbukarmaina/Beakers

Keep instruction as:

Estimate the liquid volume in milliliters. Reply with only: <number> mL.

Set Max Train / Val samples (start small if testing)

Click Load Dataset

Supported dataset fields:

image (required)

volume_ml (recommended) OR label (works too)

The trainer will automatically create labels as "<number> mL".

Step C — Train
Back to Model tab:

Batch size: 1

Grad accumulation: 4

Epochs: 3

LR: 2e-4

Click Start Training

Step D — Test inference
Open Inference tab:

Upload an image

Run inference

Output will be normalized to "<number> mL" when possible

3) Save / Upload
Save locally
In Model tab:

Choose a folder name

Click Save Model

Upload to Hugging Face
Enter username/repo-name

Paste token

Click Upload

A merged repo ..._merged will be created for easier inference.

4) Notes
If you get OOM: reduce Max Train Samples, keep 4-bit on, and reduce Max Seq Length to 128/256.

For best evaluation later, keep labels strictly numeric "<number> mL".

yaml
Copy code

---

# What you do next (3 steps)

1) Replace your current files with the ones above.
2) Run:
```bash
pip install -r requirements.txt
python app.py --share