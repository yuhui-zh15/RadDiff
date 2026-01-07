
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
import datetime
import os
import hashlib
import pickle
from collections import defaultdict
import logging
import json
import os
import csv 
import random
from openai import OpenAI
from tqdm import tqdm
import base64
from rank_bm25 import BM25Okapi
import stanza
import lmdb
from prompts import *



logging.basicConfig(filename='benchmark.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


LLM_CACHE_FILE =  "cache/cache_llm"

if not os.path.exists(LLM_CACHE_FILE):
    os.makedirs(LLM_CACHE_FILE)

llm_cache = lmdb.open(LLM_CACHE_FILE, map_size=int(1e11))

STANZA_NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

"""
-----------------------------------------------------------
Cache 
-----------------------------------------------------------
"""

def hash_key(key) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def get_from_cache(key: str, env: lmdb.Environment) -> Optional[str]:
    with env.begin(write=False) as txn:
        hashed_key = hash_key(key)
        value = txn.get(hashed_key.encode())
    if value:
        return json.loads(value.decode())
    return None


def save_to_cache(key: str, value: str, env: lmdb.Environment):
    with env.begin(write=True) as txn:
        hashed_key = hash_key(key)
        txn.put(hashed_key.encode(), value.encode())



"""
-----------------------------------------------------------
Data Preprocessing 
-----------------------------------------------------------
"""

def collect_txt_files(base_dir):
    txt_files = []
    for root, dirs, files in os.walk(base_dir, topdown=True):
        print(f"Checking directory: {root}")
        for name in files:
            if name.endswith('.txt'):
                txt_files.append(os.path.join(root, name))
        

    return txt_files

def unpickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def read_multiple_txt_files(files, num, seed=None):
    random.seed(seed)
    paths = []
    reports = []
    for file_path in random.sample(files, num):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                paths.append(file_path)
                reports.append(f'\n{file.read()}\n')  
    return paths, reports




"""
-----------------------------------------------------------
LLM  Functions 
-----------------------------------------------------------
"""

def propose(vlm_prompt,  model_version, image_url=None):
    data=[{"role": "user", "content": [
            {
                "type": "text",
                "text": vlm_prompt
            },
            ]}]
 

    if image_url != None:
        with open(image_url, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        data[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    key = json.dumps([model_version, data])
    cached_value = get_from_cache(key, llm_cache)

    if cached_value is not None:
        logging.debug(f"LLM Cache Hit")
        return cached_value

    response = client.chat.completions.create(
        model= model_version,
        response_format={ "type": "json_object" },
        messages=data,
    )
    output = response.choices[0].message.content

    try:
        response =  json.loads(output)
        save_to_cache(key, json.dumps(response), llm_cache)
        return response
    except json.JSONDecodeError as e:
        logging.debug(f"JSONDecodeError: {e}")
        logging.debug(f"Error in response: {output}")
        return None



def create_llm_string(reports, keep_track=False, report_indicies=None):
    string = ""
    if keep_track:
        for i in range(len(reports)):
            string+= "Report Index: "+ str(report_indicies[i]) + "\n"
            string+= "Report: \n " + reports[i] + "\n"
            string += "------------------ \n"
        return string 
    else:
        for i in range(len(reports)):
            string+= "Report " + str(i) + ": \n" 
            string+= reports[i] + "\n"
            string += "------------------ \n"
        return string 

    


"""
-----------------------------------------------------------
Generate Differences
-----------------------------------------------------------
"""  

def lemmatize_difference(difference):
        doc = STANZA_NLP(difference)
        sentence = ""
        for sent in doc.sentences:
            for word in sent.words:
                sentence+= word.lemma + " "
        return sentence



def bm25(reports, num, query):
    tokenized_corpus = [doc.split(" ") for doc in reports]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:num]
    return top_indices




def dedup_batch(batch_differences):
     prompt = create_prompt_dedup(batch_differences)
     output = propose(prompt, 'gpt-4o')
     if output is not None:
        dedub_batch_differences = list(output.values())[0]
        return dedub_batch_differences
     else:
        return batch_differences


def merge_batches(batch1, batch2):
    return batch1 + batch2



def dedup(all_differences,n_batches):
    level1_batch_size = len(all_differences) // n_batches
    dedub_batches = []
    for i in range(n_batches):
        if (level1_batch_size % n_batches != 0) and (i == n_batches - 1):
             batch_differences = all_differences[i * level1_batch_size : ]
        else:
            batch_differences = all_differences[i * level1_batch_size : (i + 1) * level1_batch_size]
        dedup_result = dedup_batch(batch_differences)
        if dedup_result is not None:
            dedub_batches.append(dedup_result)
        
    
    while len(dedub_batches)  > 1:
        merged_batches = []
        # If not odd number of batches, then do extra merge
        if len(dedub_batches) % 2 != 0:
            dedub_batches[1] = merge_batches(dedub_batches[0], dedub_batches[1])
            dedub_batches.pop(0)
        
        for i in range(0, len(dedub_batches), 2):
            merged_batch = merge_batches(dedub_batches[i], dedub_batches[i + 1])
            dedup_result = dedup_batch(merged_batch)
            if dedup_result is not None:
                merged_batches.append(dedup_result)
        
        dedub_batches = merged_batches
    return dedub_batches[0]

    
def create_differences(n, all_reports, num_batches):
    per_batch = n // num_batches
    all_differences = []
    for i in range(num_batches):
        _, batch_reports = read_multiple_txt_files(all_reports, per_batch, i)
        prompt = create_vlm_prompt_report(per_batch, create_llm_string(batch_reports, False))
        differences = propose(prompt, 'gpt-4o')
        if differences is None:
            continue
        all_differences.extend(list(differences.values())[0])
    return all_differences





"""
-----------------------------------------------------------
Classify Differences
-----------------------------------------------------------
"""   


def classify(difference, reports, report_indicies, num_batches):
    per_batch = len(reports) // num_batches
    all_classifications = []
    for i in range(num_batches):
        if (len(reports)  % num_batches != 0) and (i == num_batches - 1):
             batch_report_indicies = report_indicies[i * per_batch : ]
             batch_reports = reports[i * per_batch : ]
        else:
            batch_report_indicies = report_indicies[i * per_batch : (i + 1) * per_batch]
            batch_reports = reports[i * per_batch : (i + 1) * per_batch]
        
        reports_text = create_llm_string(batch_reports, True, batch_report_indicies)
        vlm_prompt = create_classification_stage_prompt(difference, reports_text)
        classes = propose(vlm_prompt, 'gpt-4o-mini')
        if classes is None:
            continue
        all_classifications.append(classes)
    return all_classifications



def classification_stage_main(diff, all_reports, n_reports_classification, num_batches):
    conditions = list(diff.values())
    conditionA = lemmatize_difference(conditions[0])
    conditionB = lemmatize_difference(conditions[1])
    
    paths, reports = read_multiple_txt_files(all_reports, len(all_reports), 42)
    top_groupA = list(bm25(reports, n_reports_classification, conditionA))
    top_groupB = list(bm25(reports, n_reports_classification, conditionB))
    top_indices = list(set(top_groupA + top_groupB))
    top_indices = top_indices[:min(len(top_indices), n_reports_classification)]
    
    paths_info = {}
    for i in top_indices:
        paths_info[i] = paths[i]
    reports = [reports[i] for i in top_indices]
    condition = f"{conditionA} vs {conditionB}"

    batches = classify(condition, reports, list(paths_info.keys()), num_batches)
    running_total = 0
    
    for x in batches:
        x['condition'] = condition
        count = 0
        batch_total = len(x['group A']) + len(x['group B']) + len(x['neither'])
        running_total += batch_total
        for group in ['group A', 'group B', 'neither']:
            for i, report in enumerate(x[group]):
                try:
                    report['report_path'] =  paths_info[int(report['report_index'])]
                    report.pop('report_index')
                except Exception:
                    count = count + 1
                    x[group].pop(i)   
            logging.info(f"{group} Length: {len(x[group])}")    
        logging.info(f"Incorrect Path Indicies: {count}/{batch_total}")  

    logging.info(f"LLM Total Classified Reports: {running_total} / {len(reports)}")
    return batches
    


"""
-----------------------------------------------------------
CSV Postprocessing
-----------------------------------------------------------
""" 


def create_differences_csv(differences, filename):
    groupA = [list(diff.values())[0] for diff in differences]
    groupB = [list(diff.values())[1] for diff in differences]
    data = {
    'condition A': groupA,
    'condition B': groupB,
    'useful (Y/N)': [''] * len(groupB),
    'difficulty (1-5)':  [''] * len(groupB)
  }

    df = pd.DataFrame(data)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = f"{filename}_{timestamp}.csv"
    df.to_csv(csv_path, index=True)

    print(f"CSV file '{csv_path}' created successfully.")



def transform_classification_output(classification_output):
    headers = ["condition", "group", "report_path", "reasoning", "direct_evidence"]
    data = []

    for difference in classification_output:
        for batch in difference:
            try:
                 condition = batch['condition']
            except Exception as e:
                logging.error(f"KeyError accessing 'condition': {e}")
                continue
            

            # Collect all report paths and their corresponding groups
            all_reports = {}
            for group in ['group A', 'group B', 'neither']:
                count = 0
                for i, report in enumerate(batch[group]):
                    try:
                        report_path = report['report_path']
                    except Exception as esc:
                        count = count + 1 
                        print(f"{esc}:{report}")
                        batch[group].pop(i)
                        continue
                    if report_path not in all_reports:
                        all_reports[report_path] = []
                    all_reports[report_path].append(group)

                logging.info("Report Path Key Error: %s/%s", str(count), str(len(batch[group])))


            # Filter out reports that are in multiple groups
            unique_reports = {k: v for k, v in all_reports.items() if len(v) == 1}

            for group in ['group A', 'group B', 'neither']:
                count = 0
                for report in batch[group]:
                    report_path = report['report_path']
                    if report_path in unique_reports and unique_reports[report_path][0] == group:
                        try:
                            reasoning = report['reasoning']
                            evidence = report['direct_evidence']
                            data.append([condition, group, report_path, reasoning, evidence])
                        except Exception as e:
                            count = count + 1
                            continue 
                logging.info("Reasoning/Direct Evidence Key Error: %s/%s", str(count), str(len(batch[group])))
    df = pd.DataFrame(data, columns=headers)
    return df



def create_verification_csv(df, N):
    random_rows = df.sample(n=N, random_state=42)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = f'classification_verification_{timestamp}.csv'
    


    def replace_with_report_content(row):
        report_path = row['report_path']
        try:
            with open(report_path, 'r') as file:
                report_content = file.read()
        except Exception as e:
            report_content = str(e)
        row['report'] = report_content
        return row


    updated_rows = random_rows.apply(replace_with_report_content, axis=1)
    updated_rows = updated_rows.drop(columns=['report_path', 'reasoning', 'direct_evidence'])
    updated_rows = updated_rows.rename(columns={'condition': 'condition ("A vs B")'})
    updated_rows['accurate (Y/N)'] = np.nan
    updated_rows['difficulty (1-5)'] = np.nan


    updated_rows.to_csv(output_file_path, index=False)

    print(f"Updated CSV file saved to {output_file_path}")



def get_image_paths(report_paths, global_folder_path):
    image_paths = []
    count = 0
    for report_path in report_paths:
        if os.path.exists(report_path):
            report_path = report_path.split(".txt")[0] 
            study = '/'.join(report_path.split('/')[-3:]) + '/'
            files = os.listdir(global_folder_path + study)
            image_paths.append(study + random.sample(files, 1)[0])
        else:
            count = count + 1
    logging.info("Incorrect paths: %s/%s", str(count), str(len(report_paths)))
    return image_paths


def create_paired_images_sets_csv(classfication_df):
 
    conditions = set(classfication_df['condition'])
    print("Number Unique Conditions:", len(conditions))
    data = []
    folder_path = ''

    for condition in conditions:
        group_a_reports = classfication_df[(classfication_df['condition'] == condition) & (classfication_df['group'] == 'group A')]['report_path'].tolist()
        group_b_reports = classfication_df[(classfication_df['condition'] == condition) & (classfication_df['group'] == 'group B')]['report_path'].tolist()

        difference = {}
        difference["set1"] = condition.split(' vs ')[0]
        difference["set2"] = condition.split(' vs ')[1]
        difference["set1_images"] = get_image_paths(group_a_reports, folder_path)
        difference["set2_images"] = get_image_paths(group_b_reports, folder_path)
        data.append(difference)


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'RadDiffBench_{timestamp}.jsonl'
    with open(filename, mode='w') as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + "\n")
    print(f"JSONL file saved to {filename}")






