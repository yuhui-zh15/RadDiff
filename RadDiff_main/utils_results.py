import pandas as pd
import wandb
import numpy as np
from typing import Dict, List, Tuple, Union
import os

# wandb.login()

# Set up wandb API
api = wandb.Api()


def get_logs(run_name: str) -> List[Dict]:
    """
    Retrieve logs from wandb for a given run name.
    
    Args:
        run_name: The wandb run name (can be just project name or "entity/project")
        entity: Optional entity name. If provided, will be prepended to run_name
        
    Returns:
        List of dictionaries containing run data
    """
    # Project is specified by <entity/project-name>
    runs = api.runs(run_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )
    runs_df.to_csv(run_name.split("/")[1] + ".csv")
    return runs_df.to_dict("records")


def get_metrics(
    results: List[Dict], keys: List[str] = ["acc@1", "acc@5", "acc@N"]
) -> Dict:
    """
    Calculate metrics from results.
    
    Args:
        results: List of result dictionaries
        keys: List of metric keys to calculate
        
    Returns:
        Dictionary with calculated metrics (mean, std, n)
    """
    metrics = {}
    for result in results:
        try:
            for key in keys:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result["summary"][key])
        except Exception as e:
            continue
    for key in keys:
        if key in metrics and metrics[key]:  # Check if key exists and has values
            metrics[key] = {
                "mean": np.mean(metrics[key]),
                "std": np.std(metrics[key]),
                "n": len(metrics[key]),
            }
        else:
            metrics[key] = {"mean": None, "std": None, "n": 0}
    return metrics


def display_results(run_name: str, description: str = "", entity: str = None):
    """
    Display results for a given run.
    
    Args:
        run_name: The wandb run name (can be just project name or "entity/project")
        description: Optional description for the results
        entity: Optional entity name. If provided, will be prepended to run_name
    """
    
    try:
        print("Calling get_logs")
        if entity and not "/" in run_name:
            full_run_name = f"{entity}/{run_name}"
        else:
            full_run_name = run_name
    
        print(full_run_name)

        results = get_logs(full_run_name)
        metrics = get_metrics(results)
        
        print(f"\n{'='*60}")
        print(f"Results for: {description or run_name}")
        
        print(metrics)
        
        print(f"{'='*60}\n")
        
        return metrics
        
    except Exception as e:
        print(f"Error retrieving results for {run_name}: {e}")
        return None
