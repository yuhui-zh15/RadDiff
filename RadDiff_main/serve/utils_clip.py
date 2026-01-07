import json
import logging
import os
from typing import List

import lmdb
import numpy as np
import requests

from serve.global_vars import CLIP_CACHE_FILE, CLIP_URL, CHEXZERO_URL
from serve.utils_general import get_from_cache, save_to_cache, normalize_path_for_hash

if not os.path.exists(CLIP_CACHE_FILE):
    os.makedirs(CLIP_CACHE_FILE)

clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))


def get_embeddings(inputs: List[str], model: str, modality: str) -> np.ndarray:
    input_to_embeddings = {}
    if modality == "image":
        # remove the file that ends with if '.gstmp' from the inputs
        inputs = [inp for inp in inputs if not inp.endswith(".gstmp")]

    for inp in inputs:
        # Normalize file paths for consistent caching across different mount points
        cache_key_input = normalize_path_for_hash(inp) if modality == "image" else inp
        key = json.dumps([cache_key_input, model])
        cached_value = get_from_cache(key, clip_cache)
        if cached_value is not None:
            logging.debug(f"CLIP Cache Hit")
            input_to_embeddings[inp] = json.loads(cached_value)

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]


    if len(uncached_inputs) > 0:
        url = {
            "ViT-bigG-14": CLIP_URL,
            "chexzero": CHEXZERO_URL,
        }[model]
        try:
            response = requests.post(
                url, data={modality: json.dumps(uncached_inputs)}
            ).json()
            for inp, embedding in zip(uncached_inputs, response["embeddings"]):
                input_to_embeddings[inp] = embedding
                # Normalize file paths for consistent caching across different mount points
                cache_key_input = normalize_path_for_hash(inp) if modality == "image" else inp
                key = json.dumps([cache_key_input, model])
                save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            logging.error(f"CLIP Error: {e}")
            for inp in uncached_inputs:
                input_to_embeddings[inp] = None


    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


if __name__ == "__main__":

    embeddings = get_embeddings(
        ["data/teaser.png"],
        "chexzero",
        "image",
    )
    print("Image", embeddings)
    embeddings_chexzero = get_embeddings(["HIIIIIIII", "hello world"], "chexzero", "text")
    print("Text", embeddings_chexzero)

