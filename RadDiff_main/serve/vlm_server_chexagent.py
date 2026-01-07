import logging
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

device = "cuda"
dtype = torch.float16


logging.info("Loading model... This might take a while.")
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
generation_config.max_length = 512
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)
model = model.to(device)

logging.info("Model loaded successfully!")


@app.route("/", methods=["POST"])
def interact_with_chexagent():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    if "text" not in request.form:
        return jsonify({"error": "Text not provided"}), 400

    images = [Image.open(request.files["image"]).convert("RGB")]
   
    prompt = request.form["text"]
    print(prompt)
    inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    result = processor.tokenizer.decode(output, skip_special_tokens=True)
    print("Caption:", result)
    return jsonify({"input": request.form["text"], "output": result})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8076, debug=False)
