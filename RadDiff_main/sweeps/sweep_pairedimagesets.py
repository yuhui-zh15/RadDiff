import json
import os
import random
import time
from typing import Dict, List

import click
import wandb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_results import display_results, get_logs, get_metrics


@click.command()
@click.option("--seed", default=0, type=int)
@click.option("--purity", default=1.0, type=float)
@click.option("--show-results", default=True, type=bool, help="Whether to display results at the end")
@click.option("--entity", default="", type=str, help="Wandb entity name")
def main(purity: float, seed: int, show_results: bool, entity: str):
    random.seed(0)
    root = "RadDiffBench"
    data = [json.loads(line) for line in open(f"{root}/RadDiffBench.jsonl")]

    # len(data)
    project_name = "RadDiffBench_gpt-4.1-nano_raddiff"
    for idx in range(0, len(data)):
        item = data[idx]

        cfg = f"""
project: {project_name}
seed: {seed}  # random seed

data:
  name: RadDiffBench
  group1: "{item['set1']}"
  group2: "{item['set2']}"
  purity: {purity}
"""     
        print("in loop:", idx)
        difficulty = "NA"
        cfg_dir = f"configs/sweep_raddiffbench_purity{purity}_seed{seed}"
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        cfg_file = f"{cfg_dir}/{idx}_{difficulty}.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg)
        print(f"python main.py --config {cfg_file}")
        os.system(f"python main.py --config {cfg_file}")


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
