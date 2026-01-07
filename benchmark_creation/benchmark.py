import pandas as pd
import numpy as np
from PIL import Image
import random
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
import base64
from benchmark_utils import *

DEBUG = False


all_reports = unpickle_file("radiology_reports.pkl")
print("Total Number of Reports: ", len(all_reports))


n = 1200
num_batches = 40
differences = create_differences(n, all_reports, num_batches)
print("Number of differences:", len(differences))
processed_differences = dedup(differences, 6)
print("Number of differences after deduplication: ", len(processed_differences))
create_differences_csv(processed_differences, "final_differences")


classification_output = []
num_differences = min(len(processed_differences), 150)
n_reports_classification = 700
num_batches = 140
num_workers = 32
count = 0
progress = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_diff = {
        executor.submit(classification_stage_main, diff, all_reports, n_reports_classification, num_batches): diff 
        for diff in random.sample(processed_differences, num_differences)}

    for future in concurrent.futures.as_completed(future_to_diff):
        diff = future_to_diff[future]
        progress += 1
        try:
            result = future.result()
            classification_output.append(result)
            print(f"New Difference Succesfully Processed: {progress}/{num_differences}")
        except Exception as exc:
            count = count + 1
            print(f'{diff} generated an exception: {exc}')
print("General Classification Stage Errors:", f'{count}/{num_differences}')


df = transform_classification_output(classification_output)
create_verification_csv(df, 50)
create_paired_images_sets_csv(df)
