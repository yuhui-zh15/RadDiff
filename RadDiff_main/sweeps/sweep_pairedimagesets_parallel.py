import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, List

import click
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_results import display_results, get_logs, get_metrics

def process_item(idx, item, purity, seed, dataset_name):
    project_name = "RadDiffBench_gpt-4.1-nano_raddiff-check-final"

    cfg = f"""
project: {project_name}
seed: {seed}  # random seed

data:
  name: PairedImageSets
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  purity: {purity}
"""     
    difficulty = "NA"
    cfg_dir = f"configs/sweep_raddiffbench_purity{purity}_seed{seed}"
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    cfg_file = f"{cfg_dir}/{idx}_{difficulty}.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)
    print(f"python main.py --config {cfg_file}")
    os.system(f"python main.py --config {cfg_file}")
    return project_name

@click.command()
@click.option("--seed", default=0, type=int)
@click.option("--purity", default=1.0, type=float)
@click.option("--show-results", default=True, type=bool, help="Whether to display results at the end")
@click.option("--entity", default="", type=str, help="Wandb entity name")
@click.option("--dataset", default="RadDiffBench.jsonl", type=str, help="Dataset JSONL file name")
def main(purity: float, seed: int, show_results: bool, entity: str, dataset: str):
    random.seed(0)
    root = "RadDiffBench"
    data = [json.loads(line) for line in open(f"{root}/{dataset}")]

    # Extract dataset name without .jsonl extension for config
    dataset_name = dataset.replace('.jsonl', '')

    max_workers = 5
    project_name = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(data):
            futures.append(executor.submit(process_item, idx, item, purity, seed, dataset_name))
        for future in as_completed(futures):
            try:
                project_name = future.result()  # Get the project name from the first completed task
            except Exception as e:
                print(f"Error processing dataset: {e}")


    # Display results if requested
    if show_results:
        print("\n" + "="*80)
        print("SWEEP RESULTS SUMMARY")
        print("="*80)
        
        try:
            # Display overall results for the sweep
            print(f"Retrieving results for project: {project_name}")
            display_results(project_name, f"Sweep Results (purity={purity}, seed={seed})", entity)
        except Exception as e:
            print(f"Could not retrieve sweep results: {e}")
            print("You may need to wait a few minutes for wandb to sync the results.")

if __name__ == "__main__":
    main()
