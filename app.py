import torch
from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# This is a very light model that fits in Render's free memory
model_path = "sshleifer/distilbart-cnn-6-6"

# Load with 'float16' to use 50% less RAM
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,  
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    
    # torch.inference_mode() keeps memory usage low during the calculation
    with torch.inference_mode():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('index.html', original_text=text, summary=summary)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

