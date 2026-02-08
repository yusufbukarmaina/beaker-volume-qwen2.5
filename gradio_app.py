import os
import re
import time
import random
import threading
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image as PILImage

import gradio as gr
import torch

from huggingface_hub import login
from datasets import load_dataset, Dataset

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# =========================
# Reproducibility
# =========================
random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# Globals
# =========================
model = None
tokenizer = None
trainer = None
training_stats = {}
train_ds_hf = None
val_ds_hf = None

# =========================
# Helpers
# =========================
VOL_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:mL|ml|ML)?")
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

def safe_get_pil_image(sample: Dict[str, Any]):
    """
    Returns a PIL.Image.Image or None.
    Handles: PIL image, HF Image dict {bytes/path}, and path strings.
    """
    img = sample.get("image", None)
    if img is None:
        return None

    # Already PIL
    if hasattr(img, "size") and hasattr(img, "mode"):
        return img.convert("RGB")

    # HF image dict from datasets
    if isinstance(img, dict):
        b = img.get("bytes", None)
        p = img.get("path", None)

        if b is not None:
            try:
                return PILImage.open(BytesIO(b)).convert("RGB")
            except Exception:
                return None

        if isinstance(p, str) and p.strip() and os.path.exists(p):
            try:
                return PILImage.open(p).convert("RGB")
            except Exception:
                return None

        return None

    # Path string
    if isinstance(img, str):
        if img.strip() and os.path.exists(img):
            try:
                return PILImage.open(img).convert("RGB")
            except Exception:
                return None
        return None

    return None

def normalize_volume_text(text: str) -> str:
    """Convert arbitrary text into strict '<number> mL' if a number exists."""
    if text is None:
        return ""
    text = str(text).strip()
    m = VOL_RE.search(text)
    if not m:
        return text
    return f"{m.group(1)} mL"


def choose_splits(ds_dict) -> Tuple[Optional[str], Optional[str]]:
    """Pick train and validation split names from common candidates."""
    keys = list(ds_dict.keys())
    train_candidates = ["train"]
    val_candidates = ["validation", "val", "dev", "eval", "test"]

    train_split = next((k for k in train_candidates if k in keys), None)
    val_split = next((k for k in val_candidates if k in keys), None)
    return train_split, val_split


def sample_to_messages(sample: Dict[str, Any], instruction: str) -> Optional[Dict[str, Any]]:
    img = safe_get_pil_image(sample)
    if img is None:
        return None  # ✅ never pass None/dict to collator

    # Dataset label columns
    if sample.get("volume_ml", None) is not None:
        label = f"{sample['volume_ml']} mL"
    elif sample.get("volume_label", None) is not None:
        label = normalize_volume_text(sample["volume_label"])
    else:
        return None

    if not str(label).strip():
        return None

    if not instruction or not str(instruction).strip():
        instruction = DEFAULT_INSTRUCTION

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": img},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": label}
        ]},
    ]

    out = {"messages": messages, "volume_ml": sample.get("volume_ml", None)}
    return out

def extract_number_ml(text: str) -> Optional[float]:
    """Extract first number from text. Returns float or None."""
    if text is None:
        return None
    m = NUM_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# =========================
# Status Manager
# =========================
class StatusManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.status = "Ready"
        self.progress = 0
        self.error = None
        self.model_loaded = False
        self.dataset_loaded = False
        self.model_trained = False
        self.training_mode = False

    def update(self, status: str, progress: Optional[int] = None, error: Optional[str] = None):
        with self._lock:
            self.status = status
            if progress is not None:
                self.progress = int(progress)
            if error is not None:
                self.error = error

    def set_flags(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get(self):
        with self._lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "error": self.error,
                "model_loaded": self.model_loaded,
                "dataset_loaded": self.dataset_loaded,
                "model_trained": self.model_trained,
                "training_mode": self.training_mode,
            }


status_manager = StatusManager()

# =========================
# Model Loading / Preparation
# =========================
def initialize_model_background(base_or_path: str, load_in_4bit: bool):
    global model, tokenizer
    try:
        status_manager.update("🔄 Loading vision model...", 10, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_name = base_or_path.strip() if base_or_path.strip() else "unsloth/Qwen2.5-VL-3B-Instruct"
        status_manager.update(f"🔄 Loading: {model_name}", 30)

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )

        # ✅ IMPORTANT: Do NOT call model.to("cuda") manually.
        # Unsloth/Transformers places the model correctly (4-bit/8-bit/meta init safe).
        model.eval()

        status_manager.update("✅ Model loaded!", 100)
        status_manager.set_flags(model_loaded=True, training_mode=False, error=None)

        try:
            footprint = model.get_memory_footprint() / 1e9
            print(f"✅ Model initialized. Memory footprint: {footprint:.2f} GB")
        except Exception:
            pass

    except Exception as e:
        msg = f"❌ Error loading model: {e}"
        print(msg)
        status_manager.update("❌ Model loading failed", 0, msg)
        status_manager.set_flags(model_loaded=False, training_mode=False)


def start_model_loading(base_or_path: str, load_in_4bit: bool):
    if status_manager.get()["model_loaded"]:
        return "✅ Model already loaded."
    t = threading.Thread(target=initialize_model_background, args=(base_or_path, load_in_4bit), daemon=True)
    t.start()
    return "🚀 Started loading model in background..."


def prepare_model_for_training():
    """Prepare model for training - ALWAYS returns a string."""
    global model

    st = status_manager.get()
    if not st["model_loaded"] or model is None:
        return "❌ Please load the model first!"

    try:
        status_manager.update("🔄 Configuring LoRA adapters...", 10)

        if hasattr(model, "peft_config"):
            status_manager.update("✅ Model already prepared for training.", 100)
            status_manager.set_flags(training_mode=True)
            return "✅ Model already prepared for training."

        status_manager.update("🔄 Applying LoRA configuration...", 40)

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        status_manager.update("🔄 Switching to training mode...", 80)

        try:
            FastVisionModel.for_training(model)
        except Exception as te:
            print(f"Warning: FastVisionModel.for_training failed, using model.train(): {te}")
            model.train()

        status_manager.set_flags(training_mode=True)
        status_manager.update("✅ LoRA applied. Ready for training.", 100)
        return "✅ LoRA applied. Model ready for training."

    except Exception as e:
        msg = f"❌ Error preparing model: {e}"
        status_manager.update("❌ Model preparation failed", 0, msg)
        status_manager.set_flags(training_mode=False)
        print(msg)
        return msg


# =========================
# Dataset Loading
# =========================
def load_dataset_background(dataset_repo: str, instruction: str, max_train_samples: int, max_val_samples: int):
    global train_ds_hf, val_ds_hf
    try:
        status_manager.update("🔄 Loading dataset...", 10, None)

        dataset_repo = dataset_repo.strip()
        if not dataset_repo:
            raise ValueError("Dataset repo is empty. Example: yusufbukarmaina/Beakers")

        # ✅ CRITICAL FIX: DO NOT STREAM for vision training (streaming causes image=None/dicts)
        ds = load_dataset(dataset_repo)

        train_split, val_split = choose_splits(ds)
        if train_split is None:
            raise ValueError(f"No train split found. Available splits: {list(ds.keys())}")

        status_manager.update(f"🔄 Reading split '{train_split}' ...", 30)

        # Use normal indexing (non-streaming). Respect max_train_samples.
        train_src = ds[train_split]
        n_train = min(int(max_train_samples), len(train_src))
        train_list = []
        skipped_train = 0

        for i in range(n_train):
            if (i + 1) % 100 == 0:
                status_manager.update(f"🔄 Reading train samples... ({i+1}/{n_train})", 35)

            item = sample_to_messages(train_src[i], instruction)
            if item is None:
                skipped_train += 1
                continue
            train_list.append(item)

        status_manager.update("🔄 Preparing validation samples...", 60)

        val_list = []
        skipped_val = 0

        if val_split is not None:
            val_src = ds[val_split]
            n_val = min(int(max_val_samples), len(val_src))

            for i in range(n_val):
                if (i + 1) % 50 == 0:
                    status_manager.update(f"🔄 Reading val samples... ({i+1}/{n_val})", 70)

                item = sample_to_messages(val_src[i], instruction)
                if item is None:
                    skipped_val += 1
                    continue
                val_list.append(item)
        else:
            # fallback: carve 5% from train as validation
            cut = max(1, int(0.05 * len(train_list)))
            val_list = train_list[:cut]
            train_list = train_list[cut:]

        status_manager.update("🔄 Converting to HF Dataset objects...", 85)

        train_ds_hf = Dataset.from_list(train_list)
        val_ds_hf = Dataset.from_list(val_list)

        status_manager.set_flags(dataset_loaded=True)
        status_manager.update(
            f"✅ Dataset ready! Train={len(train_ds_hf)} (skipped {skipped_train}) | "
            f"Val={len(val_ds_hf)} (skipped {skipped_val})",
            100
        )

    except Exception as e:
        msg = f"❌ Error loading dataset: {e}"
        print(msg)
        status_manager.update("❌ Dataset loading failed", 0, msg)
        status_manager.set_flags(dataset_loaded=False)
        train_ds_hf = None
        val_ds_hf = None

def start_dataset_loading(dataset_repo, instruction, max_train_samples, max_val_samples):
    t = threading.Thread(
        target=load_dataset_background,
        args=(dataset_repo, instruction, int(max_train_samples), int(max_val_samples)),
        daemon=True
    )
    t.start()
    return "🚀 Started dataset loading in background..."


# =========================
# Training
# =========================
def train_model_background(batch_size, grad_accum, epochs, lr, max_seq_length):
    global model, tokenizer, trainer, training_stats, train_ds_hf, val_ds_hf

    try:
        st = status_manager.get()

        if not st["model_loaded"] or model is None:
            status_manager.update("❌ Training failed", 0, "Load the model first.")
            status_manager.set_flags(training_mode=False)
            return

        if train_ds_hf is None or val_ds_hf is None:
            status_manager.update("❌ Training failed", 0, "Load the dataset first.")
            status_manager.set_flags(training_mode=False)
            return

        print("=" * 80)
        print("DEBUG: About to call prepare_model_for_training()")
        print("=" * 80)

        r = prepare_model_for_training()

        print(f"DEBUG: prepare_model_for_training() returned: {repr(r)} (type: {type(r).__name__})")

        # Ensure r is always a string
        if r is None:
            r = "❌ prepare_model_for_training() returned None (unexpected)."
        elif not isinstance(r, str):
            r = f"❌ prepare_model_for_training() returned non-string: {type(r).__name__}"

        print(f"DEBUG: Final r value: {repr(r)}")

        if "❌" in r:
            status_manager.update("❌ Training failed", 0, r)
            status_manager.set_flags(training_mode=False)
            return

        print("DEBUG: No error detected, continuing with training setup")
        print("=" * 80)

        status_manager.update("🔄 Setting up trainer...", 20)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_ds_hf,
            eval_dataset=val_ds_hf,
            args=SFTConfig(
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=int(grad_accum),
                warmup_steps=5,
                num_train_epochs=float(epochs),
                learning_rate=float(lr),
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=int(max_seq_length),
                save_strategy="steps",
                save_steps=300,
                gradient_checkpointing=True,
                dataloader_pin_memory=False,
            ),
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status_manager.update("🚀 Training started...", 50)

        stats = trainer.train()
        training_stats = stats

        status_manager.set_flags(model_trained=True)

        # Switch back to inference mode
        try:
            FastVisionModel.for_inference(model)
        except Exception:
            model.eval()

        status_manager.set_flags(training_mode=False)

        loss = getattr(stats, "training_loss", None)
        loss_txt = f"{loss:.4f}" if isinstance(loss, (float, int)) else "N/A"
        status_manager.update(f"🎉 Training completed! Loss={loss_txt}", 100)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()

        print("FULL TRAINING TRACEBACK:\n", tb)

        msg = (
            "❌ Training failed.\n\n"
            f"Exception repr:\n{repr(e)}\n\n"
            f"Traceback:\n{tb}"
        )

        status_manager.update("❌ Training failed", 0, msg)
        status_manager.set_flags(training_mode=False)


def start_training(batch_size, grad_accum, epochs, lr, max_seq_length):
    st = status_manager.get()
    if not st["model_loaded"]:
        return "❌ Please load the model first!"
    if train_ds_hf is None or val_ds_hf is None:
        return "❌ Please load the dataset first!"
    t = threading.Thread(
        target=train_model_background,
        args=(batch_size, grad_accum, epochs, lr, max_seq_length),
        daemon=True
    )
    t.start()
    return "🚀 Started training in background..."


# =========================
# Inference
# =========================
DEFAULT_INSTRUCTION = "Estimate the liquid volume in milliliters. Reply with only: <number> mL."

def run_inference(image, instruction, temperature=0.8, min_p=0.1, max_tokens=64):
    global model, tokenizer
    st = status_manager.get()

    if not st["model_loaded"] or model is None:
        return "❌ Please load the model first!"
    if image is None:
        return "❌ Please upload an image first!"
    if not instruction or not instruction.strip():
        instruction = DEFAULT_INSTRUCTION

    try:
        start = time.time()

        # Ensure inference mode
        if st["training_mode"]:
            try:
                FastVisionModel.for_inference(model)
            except Exception:
                model.eval()
            status_manager.set_flags(training_mode=False)
        else:
            model.eval()

        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image},
        ]}]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                use_cache=True,
                temperature=float(temperature),
                min_p=float(min_p),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        reply_norm = normalize_volume_text(reply)

        dt = time.time() - start
        return f"{reply_norm}\n\nInference time: {dt:.2f}s"

    except Exception as e:
        return f"❌ Error during inference: {e}"


# =========================
# Evaluation
# =========================
import numpy as np

def evaluate_model(split_choice: str, max_samples: int, temperature: float, min_p: float, max_tokens: int):
    global model, tokenizer, train_ds_hf, val_ds_hf

    st = status_manager.get()
    if not st["model_loaded"] or model is None:
        return "❌ Load the model first."
    if train_ds_hf is None or val_ds_hf is None:
        return "❌ Load the dataset first (Dataset tab)."

    split_choice = (split_choice or "val").lower().strip()
    if split_choice in ["val", "validation", "dev", "eval", "test"]:
        ds = val_ds_hf
        split_name = "val"
    else:
        ds = train_ds_hf
        split_name = "train"

    n = min(int(max_samples), len(ds))
    if n < 5:
        return "❌ Not enough samples to evaluate."

    # Ensure inference mode
    if st["training_mode"]:
        try:
            FastVisionModel.for_inference(model)
        except Exception:
            model.eval()
        status_manager.set_flags(training_mode=False)
    else:
        model.eval()

    preds, gts = [], []
    skipped = 0
    instruction = DEFAULT_INSTRUCTION

    for i in range(n):
        item = ds[i]
        messages = item.get("messages", None)
        if not messages:
            skipped += 1
            continue

        img = None
        for c in messages[0]["content"]:
            if c.get("type") == "image":
                img = c.get("image")
                break
        if img is None:
            skipped += 1
            continue

        gt = item.get("volume_ml", None)
        if gt is None:
            label_text = ""
            for c in messages[1]["content"]:
                if c.get("type") == "text":
                    label_text = c.get("text", "")
                    break
            gt = extract_number_ml(label_text)
        if gt is None:
            skipped += 1
            continue

        inf_messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": img},
        ]}]
        input_text = tokenizer.apply_chat_template(inf_messages, add_generation_prompt=True)
        inputs = tokenizer(img, input_text, add_special_tokens=False, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                min_p=float(min_p),
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        gen = out[0][input_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()

        pred = extract_number_ml(text)
        if pred is None:
            skipped += 1
            continue

        preds.append(float(pred))
        gts.append(float(gt))

    if len(preds) < 5:
        return f"❌ Too many invalid predictions. Valid={len(preds)}, Skipped={skipped}."

    preds = np.array(preds, dtype=float)
    gts = np.array(gts, dtype=float)
    err = preds - gts

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gts - np.mean(gts)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    acc_1 = float(np.mean(np.abs(err) <= 1.0) * 100.0)
    acc_2 = float(np.mean(np.abs(err) <= 2.0) * 100.0)
    acc_5 = float(np.mean(np.abs(err) <= 5.0) * 100.0)

    return (
        f"📏 Evaluation split: {split_name}\n"
        f"✅ Valid samples: {len(preds)} | Skipped: {skipped}\n\n"
        f"MAE:  {mae:.3f} mL\n"
        f"RMSE: {rmse:.3f} mL\n"
        f"R²:   {r2:.4f}\n\n"
        f"Accuracy within ±1 mL: {acc_1:.1f}%\n"
        f"Accuracy within ±2 mL: {acc_2:.1f}%\n"
        f"Accuracy within ±5 mL: {acc_5:.1f}%\n"
    )


# =========================
# Save / Upload
# =========================
def save_model(output_path):
    global model, tokenizer
    st = status_manager.get()
    if not st["model_trained"]:
        return "❌ Please complete training first!"
    try:
        output_path = output_path.strip() or "trained_beaker_model"
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        return f"✅ Model saved to: {output_path}"
    except Exception as e:
        return f"❌ Error saving model: {e}"


def upload_to_hub(model_name, hf_token, private_repo=True):
    global model, tokenizer
    st = status_manager.get()
    if not st["model_trained"]:
        return "❌ Please complete training first!"
    try:
        model_name = model_name.strip()
        if not model_name:
            return "❌ Enter a model name like: username/beaker-volume-qwen2.5vl"

        login(hf_token.strip())

        model.push_to_hub(model_name, token=hf_token, private=private_repo)
        tokenizer.push_to_hub(model_name, token=hf_token, private=private_repo)

        merged_name = f"{model_name}_merged"
        model.push_to_hub_merged(
            merged_name,
            tokenizer,
            save_method="merged_16bit",
            token=hf_token
        )

        return (
            "✅ Upload complete!\n"
            f"- LoRA model: {model_name}\n"
            f"- Merged model: {merged_name}\n\n"
            "Note: If your merged repo complains about missing configs, copy these from the base model repo:\n"
            "- generation_config.json\n"
            "- model.safetensors.index.json\n"
        )
    except Exception as e:
        return f"❌ Error uploading model: {e}"


# =========================
# Status UI
# =========================
def get_current_status():
    st = status_manager.get()
    txt = (
        f"📊 **Status**: {st['status']}\n"
        f"📈 **Progress**: {st['progress']}%\n"
        f"🤖 **Model Loaded**: {'✅' if st['model_loaded'] else '❌'}\n"
        f"📚 **Dataset Loaded**: {'✅' if st['dataset_loaded'] else '❌'}\n"
        f"🎓 **Model Trained**: {'✅' if st['model_trained'] else '❌'}\n"
        f"🏃 **Training Mode**: {'✅' if st['training_mode'] else '❌'}"
    )
    if st["error"]:
        txt += f"\n\n❌ **Error**: {st['error']}"
    return txt


# =========================
# Gradio App
# =========================
SAMPLE_INSTRUCTIONS = [
    DEFAULT_INSTRUCTION,
    "What is the liquid volume? Reply with only: <number> mL.",
    "Estimate the volume in mL. Output format: <number> mL.",
]


def create_app():
    with gr.Blocks(title="Beaker Volume Recognition", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧪 Beaker Volume Recognition (Qwen2.5-VL + Unsloth)")
        gr.Markdown("Fine-tune and test a vision-language model to estimate liquid volume in **mL**.")

        status_display = gr.Markdown(get_current_status())

        with gr.Tabs():
            with gr.TabItem("🤖 Model"):
                with gr.Row():
                    with gr.Column():
                        base_or_path = gr.Textbox(
                            label="Base / Custom Model (HF repo or local path)",
                            value="unsloth/Qwen2.5-VL-3B-Instruct"
                        )
                        load_4bit = gr.Checkbox(value=True, label="Load in 4-bit (recommended for JarvisLab)")

                        load_btn = gr.Button("🚀 Load Model", variant="primary")
                        prep_btn = gr.Button("⚙️ Prepare LoRA for Training")

                        op_status = gr.Textbox(label="Operation Status", lines=6)

                    with gr.Column():
                        gr.Markdown("### Training Hyperparameters")
                        batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                        grad_accum = gr.Slider(1, 16, value=4, step=1, label="Gradient Accumulation")
                        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                        learning_rate = gr.Slider(1e-5, 5e-4, value=2e-4, label="Learning Rate")
                        max_seq_length = gr.Slider(128, 512, value=256, step=64, label="Max Seq Length")

                        train_btn = gr.Button("🎯 Start Training", variant="primary")

                        gr.Markdown("### Save / Upload")
                        save_path = gr.Textbox(value="trained_beaker_model", label="Local Save Path")
                        save_btn = gr.Button("💾 Save Model")

                        with gr.Row():
                            hf_model_name = gr.Textbox(label="HF Model Name", placeholder="username/beaker-volume-qwen2.5vl")
                            hf_token = gr.Textbox(label="HF Token", type="password")
                        private_repo = gr.Checkbox(value=True, label="Private repo")
                        upload_btn = gr.Button("☁️ Upload to Hub")

            with gr.TabItem("📚 Dataset"):
                with gr.Column():
                    dataset_repo = gr.Textbox(
                        label="Hugging Face Dataset Repo",
                        placeholder="e.g., yusufbukarmaina/Beakers",
                        value=""
                    )
                    instruction = gr.Textbox(
                        label="Training Instruction (Prompt)",
                        value=DEFAULT_INSTRUCTION
                    )
                    max_train = gr.Slider(100, 20000, value=4000, step=100, label="Max Train Samples")
                    max_val = gr.Slider(50, 5000, value=1000, step=50, label="Max Val Samples")
                    load_ds_btn = gr.Button("📥 Load Dataset", variant="primary")
                    ds_status = gr.Textbox(label="Dataset Load Status", lines=4)

            with gr.TabItem("🔬 Inference"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Sample Instructions")
                        sample_buttons = []
                        for s in SAMPLE_INSTRUCTIONS:
                            b = gr.Button(s[:40] + ("..." if len(s) > 40 else ""), size="sm")
                            sample_buttons.append((b, s))

                    with gr.Column(scale=2):
                        image_input = gr.Image(type="pil", label="Upload Beaker Image")
                        instruction_input = gr.Textbox(label="Instruction", value=DEFAULT_INSTRUCTION)

                        with gr.Row():
                            temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
                            min_p = gr.Slider(0.01, 0.5, value=0.1, label="Min P")
                            max_tokens = gr.Slider(16, 256, value=64, step=16, label="Max Tokens")

                        with gr.Row():
                            infer_btn = gr.Button("🔬 Run Inference", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear")

                        output = gr.Textbox(label="Model Output", lines=6)

            with gr.TabItem("📏 Evaluation"):
                gr.Markdown("Run numeric evaluation (MAE/RMSE/R²) on the loaded dataset split.")

                split_choice = gr.Dropdown(choices=["val", "train"], value="val", label="Which split to evaluate?")
                eval_samples = gr.Slider(20, 2000, value=200, step=20, label="Max evaluation samples")

                with gr.Row():
                    eval_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature (lower = more stable)")
                    eval_min_p = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Min P")
                    eval_max_tokens = gr.Slider(16, 128, value=32, step=16, label="Max Tokens")

                eval_btn = gr.Button("📏 Run Evaluation", variant="primary")
                eval_out = gr.Textbox(label="Evaluation Metrics", lines=12)

        # Events
        load_btn.click(start_model_loading, inputs=[base_or_path, load_4bit], outputs=[op_status])
        prep_btn.click(prepare_model_for_training, outputs=[op_status])

        load_ds_btn.click(
            start_dataset_loading,
            inputs=[dataset_repo, instruction, max_train, max_val],
            outputs=[ds_status]
        )

        train_btn.click(
            start_training,
            inputs=[batch_size, grad_accum, epochs, learning_rate, max_seq_length],
            outputs=[op_status]
        )

        save_btn.click(save_model, inputs=[save_path], outputs=[op_status])
        upload_btn.click(upload_to_hub, inputs=[hf_model_name, hf_token, private_repo], outputs=[op_status])

        infer_btn.click(run_inference, inputs=[image_input, instruction_input, temp, min_p, max_tokens], outputs=[output])
        clear_btn.click(lambda: (None, ""), outputs=[image_input, output])

        for b, s in sample_buttons:
            b.click(lambda x=s: x, outputs=[instruction_input])

        eval_btn.click(
            evaluate_model,
            inputs=[split_choice, eval_samples, eval_temp, eval_min_p, eval_max_tokens],
            outputs=[eval_out]
        )

        # Auto refresh status
        timer = gr.Timer(value=2)
        timer.tick(get_current_status, outputs=[status_display])

    return app


# ✅ This is what app.py imports
app = create_app()

if __name__ == "__main__":
    print("🚀 Starting Beaker Volume Recognition...")
    app.launch(server_name="0.0.0.0", server_port=6006, share=True, debug=False)
