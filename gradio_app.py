import os
import gradio as gr
import torch
from huggingface_hub import HfApi
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from config import HF_DATASET_ID, QWEN_MODEL_ID, FLORENCE_MODEL_ID, QUESTION
from metrics import extract_number

STATE = {"processor": None, "tokenizer": None, "model": None, "loaded": ""}

def list_my_model_repos():
    api = HfApi()
    me = api.whoami()["name"]
    models = api.list_models(author=me)
    repos = sorted({m.modelId for m in models})
    return gr.Dropdown(choices=repos, value=(repos[0] if repos else None))

def load_model(source, family, my_repo, local_path):
    model_id = None
    if source == "Base pretrained":
        model_id = QWEN_MODEL_ID if family == "qwen" else FLORENCE_MODEL_ID
    elif source == "My HF repo":
        if not my_repo:
            raise gr.Error("Click 'Refresh my repos' then select a repo.")
        model_id = my_repo
    else:
        if not local_path:
            raise gr.Error("Provide local folder path.")
        model_id = local_path

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if family == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
    else:
        tokenizer = processor.tokenizer
        model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True, device_map="auto")

    model.eval()
    STATE["processor"], STATE["tokenizer"], STATE["model"] = processor, tokenizer, model
    STATE["loaded"] = model_id
    return f"✅ Loaded: {model_id}"

@torch.no_grad()
def predict(image):
    if STATE["model"] is None:
        return "❌ Load a model first", None

    enc = STATE["processor"](images=image, text=QUESTION, return_tensors="pt").to(STATE["model"].device)
    out = STATE["model"].generate(**enc, max_new_tokens=16)
    txt = STATE["tokenizer"].decode(out[0], skip_special_tokens=True)
    vol = extract_number(txt)
    return txt, vol

with gr.Blocks() as demo:
    gr.Markdown("# 🧪 Beaker Volume Predictor (mL)\nUpload image → predict volume.\n\nDataset is fixed (no URL paste).")

    gr.Textbox(label="HF Dataset (read-only)", value=HF_DATASET_ID, interactive=False)

    family = gr.Radio(["qwen", "florence"], value="qwen", label="Model family")
    source = gr.Radio(["Base pretrained", "My HF repo", "Local folder"], value="Base pretrained", label="Load from")

    with gr.Row():
        my_repo = gr.Dropdown(label="My HF repos", choices=[], interactive=True)
        refresh = gr.Button("🔄 Refresh my repos")
    local_path = gr.Textbox(label="Local folder path", placeholder="./outputs/qwen2_5_vl_lora")

    status = gr.Textbox(label="Status", interactive=False, value="Not loaded")
    load_btn = gr.Button("✅ Load model")

    refresh.click(list_my_model_repos, outputs=my_repo)
    load_btn.click(load_model, inputs=[source, family, my_repo, local_path], outputs=status)

    gr.Markdown("## Predict")
    img = gr.Image(type="pil", label="Upload beaker image")
    btn = gr.Button("🔍 Predict volume")
    raw = gr.Textbox(label="Model raw output")
    vol = gr.Number(label="Predicted volume (mL)")

    btn.click(predict, inputs=img, outputs=[raw, vol])

demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
