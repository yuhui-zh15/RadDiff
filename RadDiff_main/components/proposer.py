import hashlib
import json
import os
import re
import random
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
from PIL import Image

import components.prompts as prompts
import wandb
import nltk
from nltk.corpus import stopwords
from serve.utils_general import save_data_diff_image, save_data_diff_image_separate, save_data_diff_image_combined, crop_and_compose_from_original, parse_iterative_crop_coords, normalize_path_for_hash
from serve.utils_llm import get_llm_output, get_llm_output_single_image, get_llm_output_multiple_images, get_llm_output_images, get_llm_output_coordinates_unified
from serve.utils_vlm import get_embed_caption_blip, get_vlm_output
from sklearn.feature_extraction.text import TfidfVectorizer

# Download once
#nltk.download('stopwords')
load_dotenv()

def add_image_references(prev_results):
    """Add 'this difference corresponds to image X' after each Top X Score line in prev_results"""
    if not prev_results:
        return prev_results

    lines = prev_results.strip().split('\n')
    processed_lines = []
    pending_image_ref = None

    for line in lines:
        if line.startswith('Top ') and 'Difference:' in line:
            # Extract the number from "Top X Difference:"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    top_num = parts[1]
                    pending_image_ref = f"This difference corresponds to image {top_num}"
                except (ValueError, IndexError):
                    pending_image_ref = None

            line = line + f" - this difference corresponds to image {top_num}"
            processed_lines.append(line)
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)

class Proposer:
    def __init__(self, args: Dict):
        self.args = args

    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict],
    iterate:bool = False, prev_results:str = None, iteration_count: int = 1) -> Tuple[List[str], List[Dict], List[Dict]]:
        
        """
        Given two datasets, return a list of hypotheses
        """
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        
        # # Filter out .gstmp files before sampling
        # dataset1 = [item for item in dataset1 if not item["path"].endswith(".gstmp")]
        # dataset2 = [item for item in dataset2 if not item["path"].endswith(".gstmp")]
            
        for i in range(self.args["num_rounds"]): 
            print("Length of dataset1: ", len(dataset1))
            print("Length of dataset2: ", len(dataset2))
            print("Number of samples: ", self.args["num_samples"])
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            sampled_dataset2 = self.sample(dataset2, self.args["num_samples"])
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2, iterate, prev_results, iteration_count)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)

        print("Len ALL HYPOTHESES: ", len(all_hypotheses))
        print("ALL HYPOTHESES: ", all_hypotheses)

        # Apply post-processing if enabled in config (before deduplication)
        if self.args.get("post_processing", False):
            post_processing_hypotheses = self.apply_post_processing(all_hypotheses)
            print("Len POST PROCESSING HYPOTHESES: ", len(post_processing_hypotheses))
            print("Final POST PROCESSING LIST", post_processing_hypotheses)
            # Use post-processed hypotheses for deduplication
            hypotheses_to_dedup = post_processing_hypotheses
        else:
            hypotheses_to_dedup = all_hypotheses

        # Apply deduplication
        hypotheses_string = ""
        for i, hypothesis in enumerate(hypotheses_to_dedup):
            hypotheses_string += f"{i + 1}. {hypothesis}\n"
        prompt = getattr(prompts, "DEDUP_DIFF_PROMPT").format(differences = hypotheses_string)
        output = get_llm_output(prompt, self.args["model"], True)
        output = json.loads(output)
        print("Len HYPOTHESES TO DEDUP: ", len(hypotheses_to_dedup))
        if isinstance(output, str):
            output = ''.join(output)
            output = json.loads(output)
            if isinstance(output, str):
                dedub_hypotheses = [line.replace("* ", "") for line in output.splitlines()]
            else:
                dedub_hypotheses = list(output.values())[0]
        else:
            dedub_hypotheses = list(output.values())[0]
        
        dedub_hypotheses = [
            h for h in dedub_hypotheses
            if not (isinstance(h, (int, float)) or 
                   (isinstance(h, str) and h.strip().isdigit()) or
                   (isinstance(h, list) and all(isinstance(x, (int, float)) for x in h)))
        ]
        
        print("Len DEDUB HYPOTHESES: ", len(dedub_hypotheses))
        print("Final POST DEDUB LIST", dedub_hypotheses)

        return dedub_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict], iterate:bool = False, prev_results:str = None, iteration_count: int = 1
    ) -> Tuple[List[str], Dict]:
        raise NotImplementedError

    def sample(self, dataset: List[Dict], n: int) -> List[Dict]:
        return random.sample(dataset, n)

    def visualize(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Dict:
        images1 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset1
        ]
        images2 = [
            wandb.Image(
                Image.open(item["path"]).convert("RGB").resize((224, 224)),
                caption=item.get("caption", ""),
            )
            for item in sampled_dataset2
        ]
        images = {"images_group_1": images1, "images_group_2": images2}
        return images

    def captioning(self, dataset: List[Dict]):
        prompt =  getattr(prompts, self.args["captioner"]["prompt"])
        for item in dataset:
            item["caption"] = get_vlm_output(
                item["path"],
                prompt,
                self.args["captioner"]["model"],
            )

    def apply_post_processing(self, hypotheses: List[str]) -> List[str]:
        """
        Apply post-processing to the hypotheses using the POST_PROCESSING_PROMPT
        """
        hypotheses_string = ""
        for i, hypothesis in enumerate(hypotheses):
            hypotheses_string += f"{i + 1}. {hypothesis}\n"
        
        prompt = getattr(prompts, "POST_PROCESSING_PROMPT").format(differences=hypotheses_string)
        output = get_llm_output(prompt, self.args["model"], True)
        output = json.loads(output)
        
        if isinstance(output, str):
            output = ''.join(output)
            output = json.loads(output)
            if isinstance(output, str):
                post_processed_hypotheses = [line.replace("* ", "") for line in output.splitlines()]
            else:
                post_processed_hypotheses = list(output.values())[0]
        else:
            post_processed_hypotheses = list(output.values())[0]
        
        # Filter out any numeric-only entries
        post_processed_hypotheses = [
            h for h in post_processed_hypotheses
            if not (isinstance(h, (int, float)) or 
                   (isinstance(h, str) and h.strip().isdigit()) or
                   (isinstance(h, list) and all(isinstance(x, (int, float)) for x in h)))
        ]
        
        return post_processed_hypotheses


class LLMProposer(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = getattr(prompts, args["prompt"])
        self.meta_prompt = getattr(prompts, "META_PROMPT")


    def tfid(self, captions):

        standard_stop_words = set(stopwords.words('english'))

        
        medical_stop_words = {
                'normal', 'moderate', 'severe', 'mild', 'seen', 'noted', 
                'observed', 'impression', 'likely', 'suggests', 'compatible',
                'appears', 'findings', 'changes', 'without', 'consistent',
                'indicative', 'associated', 'typically', 'notably', 'presence',
                'absence', 'considered', 'rule', 'out'
            }


        custom_stop_words = list(standard_stop_words.union(medical_stop_words))

        vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
        tfidf_matrix = vectorizer.fit_transform(captions).todense()
        feature_names = vectorizer.get_feature_names_out()

        tfidf_sum = tfidf_matrix.sum(axis=0)
        tfidf_scores = pd.DataFrame(tfidf_sum, columns=feature_names).T
        tfidf_scores.columns = ['TF-IDF Score']
        sorted_tfidf_scores = tfidf_scores.sort_values(by='TF-IDF Score', ascending=False)
        top_10_words = sorted_tfidf_scores.head(10)
        return top_10_words.index.tolist()
            
      
        

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict], iterate:bool = False, prev_results:str = None, iteration_count: int = 1
    ) -> Tuple[List[str], Dict]:
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions1 = [
            f"Group A: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset1
        ]
        captions2 = [
            f"Group B: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset2
        ]
        caption_concat = "\n".join(captions1 + captions2)
   

        if iterate:
            pattern1 = r"acc@1:\s*([\d\.]+)"
            acc1_score = float(re.search(pattern1, prev_results).group(1))

            pattern2 = r"acc@5:\s*([\d\.]+)"
            acc5_score = float(re.search(pattern2, prev_results).group(1))

        
            print("Acc@1 score rounded: ", acc1_score)
            print("Acc@5 score rounded: ", acc5_score)

            if acc1_score == 0.0 and acc5_score == 0.0:
                prompt = getattr(prompts, self.args["prompt"] + "_ITERATIVE_LOW_SCORE").format(text = caption_concat, prev_results = prev_results)
            elif acc1_score == 1.0 or acc1_score == 0.5 or (acc1_score ==0.0 and acc5_score > 0.0):
                prompt = getattr(prompts, self.args["prompt"] + "_ITERATIVE_PARTIAL_MATCH").format(text = caption_concat, prev_results = prev_results)
            elif acc1_score == 2.0:
                prompt = getattr(prompts, self.args["prompt"] + "_ITERATIVE_FULL_MATCH").format(text = caption_concat, prev_results = prev_results)
            prompt = getattr(prompts, self.args["prompt"] + "_ITERATIVE_ALL").format(text = caption_concat, prev_results = prev_results)
        else:
            prompt = self.prompt.format(text=caption_concat)
            
        
        output = get_llm_output(prompt, self.args["model"], True)

        
        output = json.loads(output)
        if isinstance(output, str):
            output = ''.join(output)
            output = json.loads(output)
            if isinstance(output, str):
                hypotheses = [line.replace("* ", "") for line in output.splitlines()]
            else:
                hypotheses = list(output.values())[0]
        else:
            hypotheses = list(output.values())[0]

        


        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs



class RadiologyReports(Proposer):
     """
     Pass in ground-truth radiology reports from MIMIC-CXR. This approach establishes an upper bound on system performance
     """

     def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = getattr(prompts, args["prompt"])
        self.prompt_name = args["prompt"]
        self.topk = args["topk"]
        self.num_reports = args["num_samples"]
        self.num_hypotheses = args["num_hypotheses"]

     def extract_report_paths(self, filenames):
         path = Path(os.environ["reports_path"])
         radiolgy_reports = ""
         for filename in filenames:
             study = re.search(r"files/(p\d+/p\d+/s\d+)", filename).group(1)
             full_path = f"{path}/{study}.txt"
             with open(full_path, 'r') as file:
                    contents = file.read()
                    radiolgy_reports+= contents + "\n \n "

         return radiolgy_reports

     def summarize_reports(self, reports: str, group: str) -> str:
         """
         Summarize the radiology reports using the SUMMARY_PROMPT
         """
         prompt = getattr(prompts, "SUMMARY_PROMPT").format(reports=reports, num_reports=self.num_reports)
         output = get_llm_output(prompt, self.args["model"])
         output = json.loads(output)
         if isinstance(output, str):
            output = ''.join(output)
            output = json.loads(output)
            if isinstance(output, str):
                summaries = [line.replace("* ", "") for line in output.splitlines()]
            else:
                summaries = list(output.values())[0]
         else:
            summaries = list(output.values())[0]
        
         summaries = "\n \n ".join([f"Report Summary: {summary}" for summary in summaries])

         return summaries

     def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict], iterate:bool = False, prev_results:str = None, iteration_count: int = 1
    ) -> Tuple[List[str], Dict]:
        

            filenames1 = [item["path"] for item in sampled_dataset1]
            filenames2 = [item["path"] for item in sampled_dataset2]
            
            # Extract raw reports
            group_a_reports = self.extract_report_paths(filenames1)
            group_b_reports = self.extract_report_paths(filenames2)
            
            # Apply summary if enabled via args, but check for first_report flag
            if self.args.get("summary", False):
                print("Summarizing Group A reports...")
                group_a_summaries = self.summarize_reports(group_a_reports, "Group A")
                print("Summarizing Group B reports...")
                group_b_summaries = self.summarize_reports(group_b_reports, "Group B")
                print("Group A summaries:", group_a_summaries)
                print("Group B summaries:", group_b_summaries)

                summary_prompt = getattr(prompts, self.prompt_name + "_SUMMARY")

                all_summaries = f"GROUP A REPORT SUMMARIES: \n {group_a_summaries} \n \n GROUP B REPORT SUMMARIES: \n {group_b_summaries} \n\n"

                if not (self.args.get("first_report", False)):
                    prompt = summary_prompt.format(summaries=all_summaries, num_hypotheses=self.num_hypotheses)
                    print("first summary prompt", prompt)
                else:
                    all_reports = f"GROUP A REPORTS: \n {group_a_reports} \n \n GROUP B REPORTS: \n {group_b_reports} \n\n"
                    prompt = self.prompt.format(reports=all_reports, num_hypotheses=self.num_hypotheses)
                    # print("first Radiology prompt", prompt)
            else:
                all_reports = f"GROUP A REPORTS: \n {group_a_reports} \n \n GROUP B REPORTS: \n {group_b_reports} \n\n"
                prompt = self.prompt.format(reports=all_reports, num_hypotheses=self.num_hypotheses)
                # print("first Radiology prompt", prompt)
                        
        
                
            # get images 
            assert len(sampled_dataset1) == len(
                sampled_dataset2
            ), "Groups must be of equal size"
            assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
            filenames1 = [item["path"] for item in sampled_dataset1]
            filenames2 = [item["path"] for item in sampled_dataset2]
            
            # For combined image, use a hash of both datasets together
            # Normalize paths to original prefix for backward-compatible hashing
            normalized_filenames1 = [normalize_path_for_hash(f) for f in filenames1]
            normalized_filenames2 = [normalize_path_for_hash(f) for f in filenames2]
            all_filenames = normalized_filenames1 + normalized_filenames2
            combined_save_name = hashlib.sha256(json.dumps(all_filenames).encode()).hexdigest()

            # Keep original hashes for backward compatibility
            save_name1 = hashlib.sha256(json.dumps(normalized_filenames1).encode()).hexdigest()
            save_name2 = hashlib.sha256(json.dumps(normalized_filenames2).encode()).hexdigest()
            

            image_path1 = f"cache/images/{save_name1}.png"
            image_path2 = f"cache/images/{save_name2}.png"
            image_path_combined = f"cache/images/{combined_save_name}.png"

            os.makedirs(os.path.dirname(image_path1), exist_ok=True)
            os.makedirs(os.path.dirname(image_path2), exist_ok=True)

            save_data_diff_image_separate(sampled_dataset1, sampled_dataset2, image_path1, image_path2)
        
            if iterate:
                assert prev_results is not None, "prev_results must be provided for iterative mode"
                
                if self.args.get("summary", False):
                    # Use summary-specific iterative prompt if available, otherwise use regular iterative prompt
                    iterative_prompt = getattr(prompts, self.prompt_name + "_SUMMARY_ITERATIVE")
                    prompt = iterative_prompt.format(summaries=all_summaries, num_hypotheses=self.num_hypotheses, top=self.topk, prev_results=prev_results)
                    print("Iteration Prompt", prompt)
                else:
                    iterative_prompt = getattr(prompts, self.prompt_name + "_ITERATIVE")
                    prompt = iterative_prompt.format(reports=all_reports, num_hypotheses=self.num_hypotheses, top=self.topk, prev_results=prev_results)
                # print("Iteration Prompt", prompt)
            
            
            output = get_llm_output_images(prompt, self.args["model"], image_path1, image_path2, True)

            output = json.loads(output)
            if isinstance(output, str):
                output = ''.join(output)
                output = json.loads(output)
                if isinstance(output, str):
                    hypotheses = [line.replace("* ", "") for line in output.splitlines()]
                else:
                    hypotheses = list(output.values())[0]
            else:
                hypotheses = list(output.values())[0]

            
            logs = {"prompt": prompt, "output": output}
            return hypotheses, logs


            
            




class LLMProposerDiffusion(LLMProposer):
    def propose(
        self, dataset1: List[Dict], dataset2: List[Dict]
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Given two datasets, return a list of hypotheses
        """
        assert "prompt" in dataset1[0].keys(), "'prompt' column not in dataset"
        all_hypotheses = []
        all_logs = []
        all_images = []
        random.seed(self.args["seed"])
        for i in range(self.args["num_rounds"]):
            sampled_dataset1 = self.sample(dataset1, self.args["num_samples"])
            sampled_prompts = [
                item["prompt"] for item in sampled_dataset1
            ]  
            sampled_dataset2 = [
                item for item in dataset2 if item["prompt"] in sampled_prompts
            ]  
            sampled_dataset1 = sorted(
                sampled_dataset1, key=lambda k: k["prompt"]
            ) 
            sampled_dataset2 = sorted(
                sampled_dataset2, key=lambda k: k["prompt"]
            ) 
            hypotheses, logs = self.get_hypotheses(sampled_dataset1, sampled_dataset2)
            images = self.visualize(sampled_dataset1, sampled_dataset2)
            all_hypotheses += hypotheses
            all_logs.append(logs)
            all_images.append(images)
        return all_hypotheses, all_logs, all_images

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        # make sure 'prompt' is in dataset
        assert "prompt" in sampled_dataset1[0].keys(), "'prompt' column not in dataset"
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions = []
        for item1, item2 in zip(sampled_dataset1, sampled_dataset2):
            assert item1["prompt"] == item2["prompt"], "Prompt mismatch"
            prompt_a = f"Group A: {item1['caption']}".replace("\n", " ").strip()
            prompt_b = f"Group B: {item2['caption']}".replace("\n", " ").strip()
            captions += [f"\nPrompt: {item1['prompt']}\n{prompt_a}\n{prompt_b}"]
        caption_concat = "\n".join(captions)
        prompt = self.prompt.format(text=caption_concat)
        output = get_llm_output(prompt, self.args["model"])
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs


class VLMANDLLMProposer(Proposer):
    """
    RadDiff:
    Concatenate images (two image grids: one per each group) + provide captions and ask GPT-4o to find differences
    """

    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = getattr(prompts, args["prompt"])
        self.iterative_prompt = getattr(prompts, args["prompt"] + "_COMBINED_ITERATIVE_v4")
        self.topk = args["topk"]
        self.iterative_coords_prompt = getattr(prompts, "META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_ITERATIVE_COORDS_v2", None)


    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict], iterate:bool = False, prev_results:str = None, iteration_count: int = 1
    ) -> Tuple[List[str], Dict]:

        # get captions
        self.captioning(sampled_dataset1)
        self.captioning(sampled_dataset2)
        captions1 = [
            f"Group A: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset1
        ]
        captions2 = [
            f"Group B: {item['caption']}".replace("\n", " ").strip()
            for item in sampled_dataset2
        ]
        caption_concat = "\n".join(captions1 + captions2)

        # get images 
        assert len(sampled_dataset1) == len(
            sampled_dataset2
        ), "Groups must be of equal size"
        assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
        filenames1 = [item["path"] for item in sampled_dataset1]
        filenames2 = [item["path"] for item in sampled_dataset2]
        
        # For combined image, use a hash of both datasets together
        # Normalize paths to original prefix for backward-compatible hashing
        normalized_filenames1 = [normalize_path_for_hash(f) for f in filenames1]
        normalized_filenames2 = [normalize_path_for_hash(f) for f in filenames2]
        all_filenames = normalized_filenames1 + normalized_filenames2
        combined_save_name = hashlib.sha256(json.dumps(all_filenames).encode()).hexdigest()

        # Keep original hashes 
        save_name1 = hashlib.sha256(json.dumps(normalized_filenames1).encode()).hexdigest()
        save_name2 = hashlib.sha256(json.dumps(normalized_filenames2).encode()).hexdigest()
        
        # two image setting
        image_path1 = f"cache/images/{save_name1}.png"
        image_path2 = f"cache/images/{save_name2}.png"
        image_path_combined = f"cache/images/{combined_save_name}.png"

        os.makedirs(os.path.dirname(image_path1), exist_ok=True)
        os.makedirs(os.path.dirname(image_path2), exist_ok=True)

        if iteration_count == 1:
            # First iteration: original approach with two 4x5 grids
            save_data_diff_image_separate(sampled_dataset1, sampled_dataset2, image_path1, image_path2)
            prompt = self.prompt.format(text=caption_concat)
            print(prompt)
            
        elif iteration_count == 2:
            # Second iteration (iteration_count == 2): use combined 8x5 image and get first set of coordinates
            assert prev_results is not None, "prev_results must be provided for iterative mode"
            save_data_diff_image_combined(sampled_dataset1, sampled_dataset2, image_path_combined)
            # print("Previous results", prev_results)

            if self.args.get("iterative_cropping", False) and self.iterative_coords_prompt is not None:
                coord_prompt = self.iterative_coords_prompt.format(text=caption_concat, top=self.topk, prev_results=prev_results)
                # Ask the model to return JSON coordinates based on the combined image
                coord_output = get_llm_output_coordinates_unified(coord_prompt, self.args["model"], image_path_combined, True)
                print("Second iteration coord output:", coord_output)
                # print("Coord prompt", coord_prompt)
                try:
                    coord_output = json.loads(coord_output)
                    if isinstance(coord_output, str):
                        coord_output = ''.join(coord_output)
                        coord_output = json.loads(coord_output)
                except Exception:
                    coord_output = {}
                boxes_shared = parse_iterative_crop_coords(coord_output, expected_num=5)
                print("Second iteration post-processed coords:", boxes_shared)

                # crop_and_compose_from_original crops from original images and returns a list of 5 image paths
                combined_crops_path = f"cache/images/{combined_save_name}_iter{iteration_count}_crops"
                image_path_combined, boxed_output_paths = crop_and_compose_from_original(sampled_dataset1, sampled_dataset2, boxes_shared, combined_crops_path, grid_rows=8, grid_cols=5)
                # Verify cropped images were created successfully
                if isinstance(image_path_combined, list):
                    missing_files = [path for path in image_path_combined if not os.path.exists(path)]
                    if missing_files:
                        raise FileNotFoundError(f"Some cropped images were not created successfully: {missing_files}")

            prompt = self.iterative_prompt.format(text=caption_concat, top=self.topk, prev_results=prev_results)
            print("Second iteration prompt")
            print(prompt)
            
        else:
            raise ValueError(f"Unexpected iteration_count: {iteration_count}. Expected 1, 2, or 3+")

        # RadDiff: hypothetical differences with BOTH images and captions passed in 
        if iteration_count == 1 or not self.args.get("iterative_cropping", False):
            # First iteration: use two separate 4x5 images
            output = get_llm_output_images(prompt, self.args["model"], image_path1, image_path2, True)
            print("SEPARATE IMAGE PATHS", image_path1, image_path2)
        else:
            # For cropping + Second iteration onwards: image_path_combined is always a list of 5 images after cropping
            output = get_llm_output_multiple_images(prompt, self.args["model"], image_path_combined, True)
            print("MULTIPLE IMAGE PATHS", image_path_combined)
    
        output = json.loads(output)
        if isinstance(output, str):
              output = ''.join(output)
              output = json.loads(output)
              if isinstance(output, str):
                 hypotheses = [line.replace("* ", "") for line in output.splitlines()]
              else:
                 hypotheses = list(output.values())[0]
        else:
            hypotheses = list(output.values())[0]
        logs = {"prompt": prompt, "output": output}
        return hypotheses, logs
    

class GPTVLMProposer(Proposer):
    """
    Concatenate images (two image grids: one per each group) and ask GPT-4o to find differences.
    """
    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = getattr(prompts, args["prompt"])

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict],  iterate:bool = False, prev_results:str = None
    ) -> Tuple[List[str], Dict]:
        assert len(sampled_dataset1) == len(
            sampled_dataset2
        ), "Groups must be of equal size"
        assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
        filenames1 = [item["path"] for item in sampled_dataset1]
        filenames2 = [item["path"] for item in sampled_dataset2]
        # Normalize paths to original prefix for backward-compatible hashing
        normalized_filenames1 = [normalize_path_for_hash(f) for f in filenames1]
        normalized_filenames2 = [normalize_path_for_hash(f) for f in filenames2]
        save_name1 = hashlib.sha256(json.dumps(normalized_filenames1).encode()).hexdigest()
        save_name2 = hashlib.sha256(json.dumps(normalized_filenames2).encode()).hexdigest()
        

        image_path1 = f"cache/images/{save_name1}.png"
        image_path2 = f"cache/images/{save_name2}.png"
       
        os.makedirs(os.path.dirname(image_path1), exist_ok=True)
        os.makedirs(os.path.dirname(image_path2), exist_ok=True)

        save_data_diff_image_separate(sampled_dataset1, sampled_dataset2, image_path1,image_path2)

        output = get_llm_output_images(self.prompt, self.args["model"], image_path1, image_path2, True)
        print(output)
        output = json.loads(output)
        if isinstance(output, str):
            output = ''.join(output)
            output = json.loads(output)
            if isinstance(output, str):
                hypotheses = [line.replace("* ", "") for line in output.splitlines()]
            else:
                hypotheses = list(output.values())[0]
        else:
            hypotheses = list(output.values())[0]


        logs = {"image1": image_path1, "image2": image_path2, "prompt": self.prompt, "output": output}
        return hypotheses, logs
    



class VLMProposer(Proposer):
    """
    Concatenate images and ask VLM to find differences
    """
    def __init__(self, args: Dict):
        super().__init__(args)
        self.prompt = getattr(prompts, args["prompt"])

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        assert len(sampled_dataset1) == len(
            sampled_dataset2
        ), "Groups must be of equal size"
        assert len(sampled_dataset1) <= 20, "Groups must be smaller than 20"
        filenames = [item["path"] for item in sampled_dataset1 + sampled_dataset2]
        # Normalize paths to original prefix for backward-compatible hashing
        normalized_filenames = [normalize_path_for_hash(f) for f in filenames]
        save_name = hashlib.sha256(json.dumps(normalized_filenames).encode()).hexdigest()

        image_path = f"cache/images/{save_name}.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        save_data_diff_image(sampled_dataset1, sampled_dataset2, image_path)
        output = get_vlm_output(image_path, self.prompt, self.args["model"])
        output = output.replace("</s>", " ").strip()  # remove </s> token for llava
        hypotheses = [line.replace("* ", "") for line in output.splitlines()]
        logs = {"image": image_path, "prompt": self.prompt, "output": output}
        return hypotheses, logs


class VLMFeatureProposer(Proposer):
    def __init__(self, args: Dict):
        super().__init__(args)

    def get_hypotheses(
        self, sampled_dataset1: List[Dict], sampled_dataset2: List[Dict]
    ) -> Tuple[List[str], Dict]:
        diff_caption = get_embed_caption_blip(sampled_dataset1, sampled_dataset2)
        logs = {"output": diff_caption}
        return diff_caption, logs


def test_proposers():
    dataset = pd.read_csv("data/diffusion_plates.csv")
    dataset = dataset.to_dict("records")
    dataset1 = [item for item in dataset if item["set"] == "a_plate"]
    dataset2 = [item for item in dataset if item["set"] == "a_dinner_plate"]

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "CLIP_FRIENDLY",
        "model": "gpt-4.1-nano",
        "captioner": {
            "prompt": "Describe this image",
            "model": "llava",
        },
    }

    proposer = LLMProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)

    args = {
        "num_rounds": 2,
        "num_samples": 10,
        "num_hypotheses": 10,
        "seed": 0,
        "prompt": "VLM_PROMPT",
        "model": "llava",
    }

    proposer = VLMProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)

    args = {
        "num_rounds": 1,
        "num_samples": 10,
        "seed": 0,
    }

    proposer = VLMFeatureProposer(args)
    hypotheses, _, _ = proposer.propose(dataset1, dataset2)
    print(hypotheses)


if __name__ == "__main__":
    test_proposers()
