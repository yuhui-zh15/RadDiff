import logging
from typing import Any, Dict, List, Tuple

import click
import pandas as pd
import json
import re
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm
import timeout_decorator

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    VLMFeatureProposer,
    VLMProposer,
    GPTVLMProposer,
    VLMANDLLMProposer,
    RadiologyReports,

)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker




def compare_results(metrics, prev_round_results, results_dict):
    current_acc1, current_acc5 = metrics['acc@1'], metrics['acc@5']

    prev_acc1, prev_acc5 = prev_round_results['acc@1'], prev_round_results['acc@5']
    
   
    results_dict['acc@1_same'] = results_dict.get('acc@1_same', 0)
    results_dict['acc@1_increase'] = results_dict.get('acc@1_increase', 0)
    results_dict['acc@1_decrease'] = results_dict.get('acc@1_decrease', 0)
    
    results_dict['acc@5_same'] = results_dict.get('acc@5_same', 0)
    results_dict['acc@5_increase'] = results_dict.get('acc@5_increase', 0)
    results_dict['acc@5_decrease'] = results_dict.get('acc@5_decrease', 0)
    
   
    if current_acc1 is not None and prev_acc1 is not None:
        if current_acc1 == prev_acc1:
            results_dict['acc@1_same'] += 1
        elif current_acc1 > prev_acc1:
            results_dict['acc@1_increase'] += 1
        else:
            results_dict['acc@1_decrease'] += 1
    

    if current_acc5 is not None and prev_acc5 is not None:
        if current_acc5 == prev_acc5:
            results_dict['acc@5_same'] += 1
        elif current_acc5 > prev_acc5:
            results_dict['acc@5_increase'] += 1
        else:
            results_dict['acc@5_decrease'] += 1
    
    return results_dict
    







def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/raddiff_base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    group_name = f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})'
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name=args["data"]["name"],
            group=group_name[:min(len(group_name), 128)],
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]
    print(f"Dataset1 size: {len(dataset1)}, Dataset2 size: {len(dataset2)}")

    # Filter out .gstmp files before sampling
    dataset1 = [item for item in dataset1 if not item["path"].endswith(".gstmp")]
    dataset2 = [item for item in dataset2 if not item["path"].endswith(".gstmp")]
    print(f"Dataset1 size: {len(dataset1)}, Dataset2 size: {len(dataset2)}")
        
    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names


def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict], iterate: bool = False, prev_results:str = None, iteration_count: int = 1) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)


    print("Proposer:", proposer)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2, iterate, prev_results, iteration_count)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        num_display = min(5, len(scored_hypotheses))
        for i in range(num_display):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

        
    
       

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    scores = [hypothesis["auroc"] for hypothesis in scored_hypotheses]
    if args["wandb"]:
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth": table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses], scores


def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics

TIMEOUT = 1000
@click.command()
@click.option("--config", help="config file")
@timeout_decorator.timeout(TIMEOUT)
def main(config):
    iterative_results = defaultdict(int)
    try:
        logging.info("Loading config...")
        args = load_config(config)
        print(args)
        NUM_ITERATIONS = args.get("num_iterations", 1)  # Default to 1 if not set
        ITERATE = True if NUM_ITERATIONS > 1 else False
        print(f"Number of iterations: {NUM_ITERATIONS}")
        
        logging.info("Loading data...")
        dataset1, dataset2, group_names = load_data(args)
        
        # Initialize variables to store all hypotheses across iterations
        all_hypotheses = []
        
        # Get config option for merging hypotheses from previous iterations
        merge = args.get("merge", False) 
        
        for i in range(NUM_ITERATIONS):
            print(f"{i}interation")
        
            logging.info("Proposing hypotheses...")
            if i >= 1:
                hypotheses = propose(args, dataset1, dataset2, ITERATE, prev_round_results, i + 1)
            else:
                hypotheses = propose(args, dataset1, dataset2, False, None, i + 1)
            
            # Add current round's hypotheses to the accumulated list
            all_hypotheses.extend(hypotheses)
            print(f"Current round hypotheses: {len(hypotheses)}")
            print(f"Total accumulated hypotheses: {len(all_hypotheses)}")

            logging.info("Ranking hypotheses...")
            # Determine which hypotheses to rank based on config
            if merge:
                # Rank all accumulated hypotheses (current + previous rounds)
                hypotheses_to_rank = all_hypotheses
                print(f"Ranking all accumulated hypotheses: {len(hypotheses_to_rank)}")
            else:
                # Rank only current round's hypotheses
                hypotheses_to_rank = hypotheses
                print(f"Ranking only current round hypotheses: {len(hypotheses_to_rank)}")
            
            ranked_hypotheses, scores = rank(args, hypotheses_to_rank, dataset1, dataset2, group_names)
            print(f"Ranked Hypothesis for round {i}:", ranked_hypotheses)
            logging.info("Evaluating hypotheses...")
            print(f"Evaluating {len(ranked_hypotheses)} hypotheses for round {i}")
            metrics = evaluate(args, ranked_hypotheses, group_names)
            print(f"Metrics for round {i}:", metrics)
            if i >= 1:
                iterative_results = compare_results(metrics, prev_round_metrics, iterative_results)
                print(iterative_results)

            prev_round_results = format_prev_round(args, ranked_hypotheses, scores)
            prev_round_metrics = metrics
        

    except timeout_decorator.timeout_decorator.TimeoutError:
        logging.warning(f"The entire process exceeded the timeout of {TIMEOUT} seconds, skipping this set of hypotheses.")
        return

if __name__ == "__main__":

    main()
