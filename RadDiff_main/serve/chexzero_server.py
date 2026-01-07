import json
import logging
from typing import List

import numpy as np
import torch
import clip
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
import torchvision.transforms as transforms
from tqdm import trange
# from CheXzero.preprocess_padchest import img_to_h5
# from CheXzero.train import load_clip


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 100
DEVICE = "cuda"



checkpoint = "serve/CheXzero/checkpoints/best_64_0.0001_original_35000_0.864.pt"

model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
print("loaded model")



def get_image_embeddings(image_paths: List[str]) -> List[List[float]]:
  

    for i in trange(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        images = torch.stack(
            [preprocess(Image.open(img).convert("RGB")) for img in batch]
        ).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            image_features = image_features.cpu().numpy()
            if i == 0:
                embeddings = image_features
            else:
                embeddings = np.concatenate((embeddings, image_features))

    print("encoded image")
    return embeddings.tolist()

def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    print("running")
    for i in trange(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        text = clip.tokenize(batch).to(DEVICE)  
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
            if i == 0:
                embeddings = text_features
            else:
                embeddings = np.concatenate((embeddings, text_features))
    print("encoded text")
    return embeddings.tolist()


@app.route("/", methods=["POST"])
def interact_with_clip():
    logging.info(request.form)
    print("#############", request.form)
    if "image" in request.form:
        print("request form", request.form)
        images = json.loads(request.form["image"])
        logging.info(images)
        embeddings = get_image_embeddings(images)

    if "text" in request.form:
        texts = json.loads(request.form["text"])
        logging.info(texts)
        embeddings = get_text_embeddings(texts)

    return jsonify({"embeddings": embeddings})


if __name__ == "__main__":
    logging.info("Server is running!")
    app.run(host="0.0.0.0", port=8078, debug=False)
