import json
import logging
import os
import threading
from typing import List
from openai import OpenAI
from pydantic import BaseModel, Field

import lmdb
import base64
import openai

from serve.global_vars import LLM_CACHE_FILE, VICUNA_URL
from serve.utils_general import get_from_cache, save_to_cache

logging.basicConfig(level=logging.INFO)

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))
openai.api_key = os.environ["OPENAI_API_KEY"]




def get_llm_output_images(prompt: str, model: str, image_path1:str, image_path2:str, json_output=False):
    
    client = OpenAI()
    with open(image_path1, "rb") as image_file1:
        base64_image1 = base64.b64encode(image_file1.read()).decode('utf-8')
    with open(image_path2, "rb") as image_file2:
       base64_image2 = base64.b64encode(image_file2.read()).decode('utf-8')
   
    if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant"
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}",  "detail": "high"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}",  "detail": "high"}
                        }
                    ]
                }
            ]

            # Add the JSON extraction instruction if json_output is True
            if json_output:
                messages[0]["content"].insert(1, {
                    "type": "text",
                    "text": "Extract the following information in json format {'differences': [xxx, xxx]}. Put all the information in the array"
                })
       

    else:
        messages = prompt
    key = json.dumps([model, messages])


    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    
    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
               if json_output:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={ "type": "json_object" },
                    
                    )
               else:
                   completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    
                    )
               response = completion.choices[0].message.content
           
            if json_output:
                save_to_cache(key, json.dumps(response), llm_cache)
            else:
                save_to_cache(key, response, llm_cache)

            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."


def get_llm_output_single_image(prompt: str, model: str, image_path: str, json_output=False):
    
    client = OpenAI()
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
   
    if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant"
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",  "detail": "high"}
                    }
                ]
            }
        ]

        # Add the JSON extraction instruction if json_output is True
        if json_output:
            messages[0]["content"].insert(1, {
                "type": "text",
                "text": "Extract the following information in json format {'differences': [xxx, xxx]}. Put all the information in the array"
            })
       

    else:
        messages = prompt
    key = json.dumps([model, messages])


    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    
    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
               if json_output:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={ "type": "json_object" },
                    
                    )
               else:
                   completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    
                    )
               response = completion.choices[0].message.content
           
            if json_output:
                save_to_cache(key, json.dumps(response), llm_cache)
            else:
                save_to_cache(key, response, llm_cache)

            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."



def get_llm_output(prompt: str, model: str, proposal=None) -> str:
    
    client = OpenAI()
    if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
         if proposal:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Extract the following information in json format {'differences': [xxx, xxx]}. Put all the information in the array"},
                    {"role": "user", "content": prompt},
                ]
         elif proposal==False:
               
                messages = [
                    {"role": "system", "content": "You are a helpful assistant.  Extract the following information in json format {'score': [xxx]}. Put the score in the array"},
                    {"role": "user", "content": prompt},
                ]
         else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant.  Extract the following information in json format {'output': [xxx]}. Put the answer in the array"},
                    {"role": "user", "content": prompt},
                ]

    else:
        messages = prompt
    key = json.dumps([model, messages])

    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value
    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={ "type": "json_object" },
                
                ) 
                response = completion.choices[0].message.content
            elif model == "vicuna":
                completion = openai.Completion.create(
                    model="lmsys/vicuna-7b-v1.5",
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0,  # TODO: greedy may not be optimal
                )
                response = completion["choices"][0]["text"]

            # if proposal:
            #     save_to_cache(key, json.dumps(response), llm_cache)
            # else:
            #     save_to_cache(key, response, llm_cache)
            save_to_cache(key, json.dumps(response), llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."


def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt





def get_llm_output_multiple_images(prompt: str, model: str, image_paths: List[str], json_output=False):
    """
    Get LLM output for multiple images (up to 5 cropped grid images).
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model name to use
        image_paths: List of paths to the cropped grid images
        json_output: Whether to expect JSON output
    
    Returns:
        LLM response as string
    """
    client = OpenAI()
    
    # Encode all images to base64
    base64_images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(base64_image)
    
    if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
        # Build content with text and all images
        content = [
            {"type": "text", "text": "You are a helpful assistant"},
        ]
        
        if json_output:
            content.append({
                "type": "text",
                "text": "Extract the following information in json format {'differences': [xxx, xxx]}. Put all the information in the array"
            })
        
        content.append({"type": "text", "text": prompt})
        
        # Add all images
        for i, base64_image in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
            })
        
        messages = [{"role": "user", "content": content}]
    else:
        messages = prompt
    
    key = json.dumps([model, messages, "multiple_images"])
    
    cached_value = get_from_cache(key, llm_cache)
    # cached_value = None
    if cached_value is not None:
        logging.debug(f"LLM Multiple Images Cache Hit")
        return cached_value
    
    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
                if json_output:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                else:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                response = completion.choices[0].message.content
            
            if json_output:
                save_to_cache(key, json.dumps(response), llm_cache)
            else:
                save_to_cache(key, response, llm_cache)
            
            return response
        
        except Exception as e:
            logging.error(f"LLM Multiple Images Error: {e}")
            continue
    
    return "LLM Error: Cannot get multiple images response."



def get_llm_output_coordinates_unified(prompt: str, model: str, image_paths, json_output=True):
    """
    Extract exactly 5 coordinate boxes from one or more images using structured outputs.
    This is a cleaner, unified version that handles both single and multiple images.

    Args:
        prompt: The coordinate extraction prompt
        model: The model name to use
        image_paths: Single image path (str) or list of image paths
        json_output: Always True for coordinate extraction

    Returns:
        JSON string with schema: {"boxes": [{"x1": float, "y1": float, "x2": float, "y2": float}, ...]}
    """
    # Define structured output schema using Pydantic
    class CoordinateBox(BaseModel):
        x1: float = Field(ge=0.0, le=1.0, description="Normalized left coordinate in [0,1]")
        y1: float = Field(ge=0.0, le=1.0, description="Normalized top coordinate in [0,1]")
        x2: float = Field(ge=0.0, le=1.0, description="Normalized right coordinate in [0,1]")
        y2: float = Field(ge=0.0, le=1.0, description="Normalized bottom coordinate in [0,1]")

    class CoordinatesResponse(BaseModel):
        boxes: List[CoordinateBox] = Field(
            min_length=5,
            max_length=5,
            description="Exactly 5 coordinate boxes for the top differences",
            examples=[[
                {"x1": 0.10, "y1": 0.10, "x2": 0.20, "y2": 0.20},
                {"x1": 0.30, "y1": 0.15, "x2": 0.40, "y2": 0.25},
                {"x1": 0.50, "y1": 0.20, "x2": 0.60, "y2": 0.30},
                {"x1": 0.70, "y1": 0.25, "x2": 0.80, "y2": 0.35},
                {"x1": 0.05, "y1": 0.50, "x2": 0.15, "y2": 0.60}
            ]]
        )

    client = OpenAI()

    # Convert single path to list for uniform handling
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Encode all images to base64
    base64_images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(base64_image)

    if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
        # Simple, clean structured approach
        content = [
            {"type": "text", "text": "You are a helpful assistant. You must provide exactly 5 coordinate boxes."},
            {"type": "text", "text": prompt}
        ]

        # Add all images
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
            })

        messages = [{"role": "user", "content": content}]
    else:
        messages = prompt

    # Cache key differentiates between single/multiple images
    cache_suffix = f"coordinates_unified_{len(image_paths)}images"
    key = json.dumps([model, messages, cache_suffix])

    cached_value = get_from_cache(key, llm_cache)
    if cached_value is not None:
        logging.debug(f"LLM Coordinates Unified Cache Hit ({len(image_paths)} images)")
        return cached_value

    # Simple retry with structured output only - no complex fallback logic
    for _ in range(3):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"]:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=CoordinatesResponse,
                )
                response_obj = completion.choices[0].message.parsed
                response = response_obj.model_dump_json()

            save_to_cache(key, response, llm_cache)
            return response

        except Exception as e:
            logging.error(f"LLM Coordinates Unified Error ({len(image_paths)} images): {e}")
            continue

    return "LLM Error: Cannot get coordinate response."


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-4.1-mini"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-4.1-nano"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "vicuna"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")


def test_get_llm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_llm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_get_differences():
    captions1 = [
        "A cat is sitting on a table",
        "A dog is sitting on a table",
        "A pig is sitting on a table",
    ]
    captions2 = [
        "A cat is sitting on the floor",
        "A dog is sitting on the floor",
        "A pig is sitting on the floor",
    ]
    differences = get_differences(captions1, captions2, "gpt-4")
    print(f"{differences=}")


if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_output_parallel()
    test_get_differences()
