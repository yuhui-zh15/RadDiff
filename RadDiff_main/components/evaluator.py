import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import json

from serve.utils_llm import get_llm_output


class GPTEvaluator:
    """
    Ask GPT if the hypothesis is true or false.
    """

    prompt = """
    
    I am a machine learning researcher working on identifying differences between two groups of radiology images. My goal is to determine how well a given prediction corresponds to the findings or conditions that are more commonly present in Group A compared to Group B.

    You will receive descriptions of Group A and Group B, along with a prediction. 
    
    Your task: 
    Evaluate whether the prediction is more aligned with Group A or Group B, using the following scoring system:

    2: Fully aligned with Group A.
    1: Partially aligned with Group A (i.e., the prediction is closer to Group A than Group B but represents a broader or narrower concept).
    0: Not aligned with Group A (i.e., more aligned with Group B or represents a completely different concept).
    
    Reference Example 1:
    Group A: "Left-sided opacity" and Group B: "Right-sided opacity"
    Prediction: "Left-sided opacity" → Score: 2 (fully aligned with Group A)
    Prediction: "Left lung consolidation" → Score: 2 (fully aligned with Group A)
    Prediction: "Unilateral lung opacity" → Score: 1 (broader but closer to Group A)
    Prediction: "Right-sided opacity" → Score: 0 (aligned with Group B)

    Reference Example 2:
    Group A: "Pleural effusion" and Group B: "No pleural effusion"
    Prediction: "Pleural effusion" → Score: 2 (fully aligned with Group A)
    Prediction: "Fluid in the pleural space" → Score: 2 (fully aligned with Group A)
    Prediction: "Increased fluid in the chest cavity" → Score: 1 (broader but closer to Group A)
    Prediction: "Normal lungs" → Score: 0 (aligned with Group B)


    Now, analyze the following using similar reasoning from the above examples as a guide. 

    Group A: {gt_a}
    Group B: {gt_b}
    Prediction: {hypothesis}

    Please respond with 2, 1, or 0, based on the alignment of the prediction with Group A.
    
    """
    print(prompt)
    def __init__(self, args: Dict):
        
        self.args = args

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
    
        scores = []
        evaluated_hypotheses = []
        for hypothesis in tqdm(hypotheses[: self.args["n_hypotheses"]]):
            prompt = self.prompt.format(hypothesis=hypothesis, gt_a=gt_a, gt_b=gt_b)

            answer = get_llm_output(prompt, self.args["model"], False)
        
            output = json.loads(answer)
            if isinstance(output, str):
                output = ''.join(output)
                output = json.loads(output)

            print("evaluator output", output)
            
            try: 
                output = list(output.values())[0]
                if isinstance(output, list):
                    output = output[0]   
                scores.append(int(output))
            except (ValueError, IndexError) as e:
                print("Evaluator Error: ", e)
                scores.append(0)

            evaluated_hypotheses.append(
                {"hypothesis": hypothesis, "score": scores[-1], "response": answer}
            )

        metrics = {
            "acc@1": scores[0] / 2,
            "acc@5": np.max(scores[:5]) / 2,
            "acc@N": np.max(scores[: self.args["n_hypotheses"]]) / 2,
        }
        return metrics, evaluated_hypotheses
    


class NullEvaluator:
    def __init__(self, args: Dict):
        self.args = args

    def evaluate(
        self, hypotheses: List[str], gt_a: str, gt_b: str
    ) -> Tuple[Dict, List[Dict]]:
        return {}, [{}]


def test_evaluator():
    args = {
        "model": "gpt-4.1-nano ",
        "n_hypotheses": 20,
    }
    evaluator = GPTEvaluator(args)
    hypotheses = [
        "dogs in the snow",
        "golden retrivers on a ski slope",
        "animals in the snow",
        "dogs in winter time",
    ]
    gt_a = "images of dogs in the snow"
    gt_b = "images of dogs next to cats"
    metrics, evaluated_hypotheses = evaluator.evaluate(hypotheses, gt_a, gt_b)
    print(metrics)
    print(evaluated_hypotheses)


if __name__ == "__main__":
    test_evaluator()
