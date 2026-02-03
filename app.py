import torch
from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# 1. Using a smaller model + Half-precision (Diet Mode)
model_path = "sshleifer/tinydistillbart-cnn-12-6"

# This loads the model much more efficiently for limited RAM
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Cuts memory in half!
    low_cpu_mem_usage=True      # Prevents memory spikes
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    
    # Using 'torch.inference_mode' saves even more RAM during the actual summary
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('index.html', original_text=text, summary=summary)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
