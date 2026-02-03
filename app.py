import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import pytextrank
from rouge_score import rouge_scorer
import torch

app = Flask(__name__)

# --- Manual Model Loading (No Pipelines) ---
print("--- Loading AI Models Manually (Bypassing Tasks) ---")

# 1. Extractive
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
if "textrank" not in nlp.pipe_names:
    nlp.add_pipe("textrank")

    # 2. Abstractive (BART) - Loaded directly as a model object
    print("--- AI is waking up... this takes 1 minute ---")
    model_path = "sshleifer/distilbart-cnn-6-6"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    print("--- ALL SYSTEMS READY: http://127.0.0.1:5000 ---")




@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        text = request.form.get("input_text")
        if text and len(text.split()) > 20:
            # --- Extractive Step ---
            doc = nlp(text)
            ext_sum = " ".join([sent.text for sent in doc._.textrank.summary(limit_sentences=3)])

            # --- Abstractive Step (Manual Inference) ---
            # This replaces the pipeline entirely to avoid the KeyError
            inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, do_sample=False)
            abs_sum = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # --- ROUGE Step ---
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(ext_sum, abs_sum)

            results = {
                "extractive": ext_sum,
                "abstractive": abs_sum,
                "rouge1": round(scores['rouge1'].fmeasure, 2),
                "rougeL": round(scores['rougeL'].fmeasure, 2)
            }

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)