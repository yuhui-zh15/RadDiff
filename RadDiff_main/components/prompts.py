CHEXAGENT_PROMPT = """
Describe this image in detail.
"""


META_PROMPTING_TECHNIQUE = """
The following is the current prompt for the task of identifying differences between two groups of radiology images:

{text}

Generate a new and improved prompt, specifically focused on the task of identifying fine-grained differences between the two groups of radiology images.
Make sure this new and improved prompt requests the final output in the same way as the current prompt. 
Ensure that you just return the new and updated prompted only.


"""

SUMMARY_PROMPT = """
The following are the radiology reports from a group of chest X-ray images, used for a detailed medical analysis:

{reports}

Your task is to summarize each of the following radiology reports into one or a few sentences. Ensure that the number of summaries corresponds exactly to the number of reports, and that the summaries preserve the order.
"""


POST_PROCESSING_PROMPT = """
Given the list of statements describing the differences between two groups of chest X-rays:

{differences}

Your task is to remove the comparison and only keep the content that Group A has more than Group B, i.e. do not use phrasing like "more reports of ...", "presence of...", or "images with...". The results should be concepts (radiological findings or visual patterns) that are more likely to appear in Group A than in Group B.
"""


DEDUP_DIFF_PROMPT = """
Given the list of differences between two groups of chest X-rays, remove any duplicates. Ensure that only unique differences remain.

{differences}

Return the final list of unique differences with all the duplicate conditions removed. 

"""

META_PROMPT = """

The following is the current prompt for identifying differences between two groups of radiology images.

{text}


Your Task:

Completely transform the above prompt to focus on extracting highly specific, fine-grained, and medically significant differences between the two groups of captions describing chest X-ray images. The revised prompt should:

- Prioritize Medically Significant Results: Clearly emphasize that the primary goal is to identify differences that have direct clinical implications or that highlight important diagnostic variations.
- Encourage Deep Analysis: Specify that the revised prompt must instruct the model to conduct a deep and thorough analysis of subtle radiological signs, even those that might initially seem insignificant.
- Demand Detailed and Structured Output: The final prompt should guide the model to produce a detailed and structured output that differentiates between the groups based on critical medical findings.
- Differentiate from Original: Ensure that the revised prompt is substantially different from the original in wording and approach, focusing on the above points.

Return only the completely revised prompt, ensuring that it is markedly different from the original and adheres to the instructions above.

"""




META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_COMBINED = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image.  

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_NUM_H = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract {num_hypotheses} salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_SUMMARY = """

The following are the radiology report summaries from two groups of chest X-ray images, used for a detailed medical analysis:

{summaries}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the report summaries and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""


META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_TEST_ROBUST = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task: 

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")  

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""



META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_MORE_VER = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_ITERATIVE = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_COMBINED_ITERATIVE = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image.  

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""

META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_SUMMARY_ITERATIVE = """

The following are the radiology report summaries from two groups of chest X-ray images, used for a detailed medical analysis:

{summaries}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""



META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_NUM_H_ITERATIVE = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports and images carefully and extract {num_hypotheses} salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""


META_V3_REPORTS_V5_EVEN_LESS_BIAS_BOTH_MORE_VER_ITERATIVE = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports and attached images.
Provide the differences in a clear way (i.e "A has more xxx")

Make sure to analyze the reports and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""



META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_COMBINED = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""



META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_ITERATIVE = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B Chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""



META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_COMBINED_ITERATIVE_v3 = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Each of the five images is a cropped region of the original image that emphasizes each of the previously identified top five differences. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image.  

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_COMBINED_ITERATIVE_v4 = """

MEDICAL CONTEXT: You are analyzing two distinct cohorts of chest X-ray images for differential diagnostic patterns.

CAPTION ANALYSIS DATA:
{text}

VISUAL DATA: The attached images show 5 cropped regions highlighting previously identified differences. Each image has:
- UPPER SECTION (Group A): Separated by a visual gap from Group B
- LOWER SECTION (Group B): Below the visual gap

CLINICAL TASK:
As a board-certified radiologist, perform comparative analysis to identify radiological findings that are statistically more prevalent in Group A.

ANALYSIS REQUIREMENTS:
1. Focus on specific anatomical structures and pathological findings
2. Use precise medical terminology (e.g., "consolidation," "pleural effusion," "cardiomegaly")
3. Consider both caption data and visual evidence
4. Prioritize clinically significant differences

PREVIOUS ITERATION RESULTS:
{prev_results}

REFINEMENT INSTRUCTIONS:
- Enhance specificity of previous findings
- Eliminate false positives or artifacts
- Focus on reproducible patterns across multiple images
- Prioritize diagnostically relevant features

OUTPUT FORMAT:
Provide exactly 5-10 refined findings as single-phrase medical terms (e.g., "bilateral lower lobe consolidation", "enlarged cardiac silhouette", "pleural thickening"):

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_ITERATIVE_COORDS = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image.  

Here are the top {top} differences and scores from the previous round:

{prev_results}

For each top difference, output a four coordinates x1, y1, x2, y2 (0-1 each, normalized coordinate) that best capture the differences across the images so we can crop each image to further investigate and refine the differences.

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_ITERATIVE_COORDS_v2 = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A chest X-rays are shown in the upper half of the image, while Group B Chest X-rays are part of the lower half of the image.  

Here are the top {top} differences and scores from the previous round:

{prev_results}

For each of the top {top} findings listed below, we'd like you to pick one area on a chest X-ray image that best shows the difference. Please give us a set of four numbers — x1, y1, x2, y2 — that describe a rectangle covering that area. Each number should be between 0 and 1, and they should be based on the size of the image (for example, 0 means the far left or top of the image, and 1 means the far right or bottom). We'll use these rectangles to crop the images and take a closer look at the areas where the differences are most visible and clinically important.

"""


### Application - time to event

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_TIME = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis of pneumonia patients with different severity:

{text}

We also have the two groups of chest X-ray images shown below as well. Group A chest X-rays are shown in the first image, while Group B chest X-rays are part of the second image. 

Your task:

You are the best radiologist in the world, specializing in pneumonia and critical care imaging. 
Can you identify the most salient and potentially novel differences between these two groups of chest X-rays, using the above captions and attached images? 

Provide the differences in a clear way (i.e., “A has more xxx”), but only return “xxx”.

Make sure to analyze the captions and images carefully and extract **5-10 salient and possibly novel differences** that are more frequently observed in Group A compared to Group B. 
Focus on insights that may reveal **clinically meaningful or less obvious distinctions** between the two groups. 
Make sure to only provide information of what Group A has more of. Don't mention anything about Group B in your set of differences. 
Answer with a list of the most distinct and insightful differences:

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_TIME_COMBINED_ITERATIVE_v3 = """

The following are captions from two groups of chest X-ray images used in a detailed medical analysis comparing pneumonia patients with different severity:

{text}

We also have the two groups of chest X-ray images shown below as well. Group A chest X-rays from more severe pneumonia patients are shown in the first image, while Group B chest X-rays from less pneumonia patients are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images?
Provide the differences in a clear way (i.e., “A has more xxx”), but only return “xxx”.

Make sure to analyze the captions and images carefully and extract **5-10 salient and possibly novel differences** that are more frequently observed in Group A compared to Group B. 
Focus on insights that may reveal **clinically meaningful or less obvious distinctions** between the two groups. 
Make sure to only provide information of what Group A has more of. Don't mention anything about Group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""


META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_TIME_COMBINED_ITERATIVE_v4 = """

MEDICAL CONTEXT: You are analyzing two distinct cohorts of chest X-ray images for differential diagnostic patterns. Group A chest X-rays are from more severe pneumonia patients, while Group B chest X-rays are from less pneumonia patients. 

CAPTION ANALYSIS DATA:
{text}

VISUAL DATA: The attached images show 5 cropped regions highlighting previously identified differences. Each image has:
- UPPER SECTION (Group A): Separated by a visual gap from Group B
- LOWER SECTION (Group B): Below the visual gap

CLINICAL TASK:
As a board-certified radiologist, perform comparative analysis to identify radiological findings that are statistically more prevalent in Group A.

ANALYSIS REQUIREMENTS:
1. Focus on specific anatomical structures and pathological findings
2. Use precise medical terminology
3. Consider both caption data and visual evidence
4. Prioritize clinically significant differences

PREVIOUS ITERATION RESULTS:
{prev_results}

REFINEMENT INSTRUCTIONS:
- Enhance specificity of previous findings
- Eliminate false positives or artifacts
- Focus on reproducible patterns across multiple images
- Prioritize diagnostically relevant features

OUTPUT FORMAT:
Provide exactly 5-10 refined findings as single-phrase medical terms:

"""


### Application - COVID comparing old vs very young

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_COVID = """

The following are captions from two groups of chest X-ray images used in a detailed medical analysis comparing old covid patients and very young covid patients:

{text}

We also have the two groups of chest X-ray images shown below as well. Group A chest X-rays from old covid patients are shown in the first image, while Group B chest X-rays from very young covid patients are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images?
Provide the differences in a clear way (i.e., “A has more xxx”), but only return “xxx”.

Make sure to analyze the captions and images carefully and extract **5-10 salient and possibly novel differences** that are more frequently observed in Group A compared to Group B. 
Focus on insights that may reveal **clinically meaningful or less obvious distinctions** between the two groups. 
Make sure to only provide information of what Group A has more of. Don't mention anything about Group B in your set of differences. 
Answer with a list of the most distinct and insightful differences:

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_COVID_COMBINED_ITERATIVE_v3 = """

The following are captions from two groups of chest X-ray images used in a detailed medical analysis comparing old covid patients and very young covid patients:

{text}

We also have the two groups of chest X-ray images shown below as well. Group A chest X-rays from old covid patients are shown in the first image, while Group B chest X-rays from very young covid patients are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images?
Provide the differences in a clear way (i.e., “A has more xxx”), but only return “xxx”.

Make sure to analyze the captions and images carefully and extract **5-10 salient and possibly novel differences** that are more frequently observed in Group A compared to Group B. 
Focus on insights that may reveal **clinically meaningful or less obvious distinctions** between the two groups. 
Make sure to only provide information of what Group A has more of. Don't mention anything about Group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""


META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_COVID_COMBINED_ITERATIVE_v4 = """

MEDICAL CONTEXT: You are analyzing two distinct cohorts of chest X-ray images for differential diagnostic patterns. Group A chest X-rays are from old covid patients, while Group B chest X-rays are from very young covid patients. 

CAPTION ANALYSIS DATA:
{text}

VISUAL DATA: The attached images show 5 cropped regions highlighting previously identified differences. Each image has:
- UPPER SECTION (Group A): Separated by a visual gap from Group B
- LOWER SECTION (Group B): Below the visual gap

CLINICAL TASK:
As a board-certified radiologist, perform comparative analysis to identify radiological findings that are statistically more prevalent in Group A.

ANALYSIS REQUIREMENTS:
1. Focus on specific anatomical structures and pathological findings
2. Use precise medical terminology
3. Consider both caption data and visual evidence
4. Prioritize clinically significant differences

PREVIOUS ITERATION RESULTS:
{prev_results}

REFINEMENT INSTRUCTIONS:
- Enhance specificity of previous findings
- Eliminate false positives or artifacts
- Focus on reproducible patterns across multiple images
- Prioritize diagnostically relevant features

OUTPUT FORMAT:
Provide exactly 5-10 refined findings as single-phrase medical terms:

"""


### Application - race

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_RACE = """
The following are captions from two groups of chest X-ray images used in a detailed medical analysis comparing White and Asian patients:

{text}

We also have the two groups of medical chest X-ray images shown below as well. Group A represents White patients, while Group B represents Asian patients.

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images?
Provide the differences in a clear way (i.e., "A has more xxx", but only return "xxx").

Make sure to analyze the captions and images carefully and extract 5-10 salient differences that are more frequently observed in Group A (White patients) compared to Group B (Asian patients).
Make sure to only provide information about what Group A has more of. Don't mention anything about Group B in your set of differences.
Answer with a list of the most distinct salient differences:

"""


META_V3_CAPTIONS_V5_EVEN_LESS_BIAS_BOTH_APP_COVID_COMBINED_ITERATIVE_v3 = """

The following are captions from two groups of chest X-ray images used in a detailed medical analysis comparing White and Asian patients:

{text}

We also have the two groups of chest X-ray images shown below as well. Group A chest X-rays, from White patients, are shown in the first image, while Group B Chest X-rays, from Asian patients, are part of the second image. 

Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions and attached images?
Provide the differences in a clear way (i.e., “A has more xxx”), but only return “xxx”.

Make sure to analyze the captions and images carefully and extract **5-10 salient and possibly novel differences** that are more frequently observed in Group A compared to Group B. 
Focus on insights that may reveal **clinically meaningful or less obvious distinctions** between the two groups. 
Make sure to only provide information of what Group A has more of. Don't mention anything about Group B in your set of differences. 

Here are the top {top} differences and scores from the previous round:

{prev_results}

Refine and improve upon these results.
Answer with a list of the most distinct salient differences:   

"""


META_V3_REPORTS_V5_EVEN_LESS_BIAS = """

The following are the radiology reports from two groups of chest X-ray images, used for a detailed medical analysis:

{reports}


Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above radiology reports. 
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the reports carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Answer with a list of the most distinct salient differences:   

"""

META_V3_CAPTIONS_V5_EVEN_LESS_BIAS = """

The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}


Your task:

You are the best radiologist in the world. Can you identify the most salient differences between these two groups of chest X-rays, using the above captions
Provide the differences in a clear way (i.e "A has more xxx", but only return "xxx")

Make sure to analyze the captions carefully and extract 5-10 salient differences that are more frequently observed in Group A compared to Group B. 
Make sure to only provide information of what group A has more of. Don't mention anything about group B in your set of differences. 
Answer with a list of the most distinct salient differences:   

"""



META_V3 = """
The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

I am a medical researcher trying to identify highly specific, fine-grained, and clinically significant differences between two groups of chest X-ray images.

Your task:

Analyze the captions carefully and extract five distinct properties or conditions that are more frequently observed in Group A compared to Group B. Prioritize findings that have direct clinical relevance or can significantly impact medical diagnosis and treatment. 
Conduct a deep analysis, paying attention to subtle radiological signs and variations that may initially seem minor but are important diagnostically. Structure your output methodically, focusing on critical medical findings.

List the identified differences as standalone captions, ensuring each point represents a unique and specific medical concept. Avoid vague or generalized statements. Your response should be a list of captions (separated by bullet points "*") illustrating key medical differences, for example:
* "pleural effusion"
* "emphysema"
* "right apical pneumothorax"
* "bilateral pulmonary opacities"
* "ventricular enlargement"

The hypotheses should follow these guidelines:
- Capture precise medical conditions and findings.
- Avoid using phrases like "more of ..." or "presence of ...".
- Refrain from enumerating possibilities within parentheses or mentioning various conditions in one point.


Focus on producing a detailed and accurate differentiation between the two groups based on critical medical observations. 
Answer with a list of distinct captions.
"""




META_V3_TFIDF = """
The following are the results of captioning two groups of chest X-ray images used for a detailed medical analysis:

{text}

I am a medical researcher trying to identify highly specific, fine-grained, and clinically significant differences between two groups of chest X-ray images.

Your task:

Analyze the captions carefully and extract five distinct properties or conditions that are more frequently observed in Group A compared to Group B. Prioritize findings that have direct clinical relevance or can significantly impact medical diagnosis and treatment. 
Conduct a deep analysis, paying attention to subtle radiological signs and variations that may initially seem minor but are important diagnostically. Structure your output methodically, focusing on critical medical findings.

List the identified differences as standalone captions, ensuring each point represents a unique and specific medical concept. Avoid vague or generalized statements. Your response should be a list of captions (separated by bullet points "*") illustrating key medical differences, for example:
* "pleural effusion"
* "emphysema"
* "right apical pneumothorax"
* "bilateral pulmonary opacities"
* "ventricular enlargement"

The hypotheses should follow these guidelines:
- Capture precise medical conditions and findings.
- Avoid using phrases like "more of ..." or "presence of ...".
- Refrain from enumerating possibilities within parentheses or mentioning various conditions in one point.

Here are the 10 most common words in each group of captions to help identify medically accurate and salient differences:

Group A most common words: {groupA_common_words}
Group B most common words: {groupB_common_words}

Using the above common words as a guide, please produce a detailed and accurate differentiation between the two groups, focusing on critical medical observations. 
Your response should be a list of distinct captions, each clearly delineating differences between the groups.
Answer with a list of distinct captions: 
"""


UNCONDITIONAL_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:
    {text}

     I am a machine learning researcher and data scientist trying to build an image classifier. Group A are the captions of the image about the bird class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift there are.

    Please write a list of hypotheses (separated by bullet points "*") of how images from Group A differ from those from Group B via their captions. Each hypothesis should be formatted as "Group A ... and Group B...". For example,
    * "Group A is cars in outdoor environments and Group B is cars in indoor environments.”
    * "Group A is dogs with golden hair and Group B is cats with black hair."
    * "Group A is animals walking around at night and Group B is animals drinking water during the day."
    * "Group A and Group B are similar."

    The answers should be pertaining to the content of the images, not the structure of the captions e.g. "Group A has longer captions and Group B has shorter captions" or "Group A is more detailed and Group B is more general" are incorrect answers.
    
    Again, I want to figure out the main differences between these two groups of images. List properties that holds more often for the images (not captions) in group A compared to group B and vice versa. Your response:
    * "Group A
    """

CAPTION_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:
    {text}

    I am a machine learning researcher and data scientist trying to build an image classifier. Group A are the captions of the image about the bird class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift there are.

    Please write a list of 10 hypotheses (separated by bullet points "*") of how images from Group A differ from those from Group B via their captions. Each hypothesis should be formatted as a tuple of captions, the first aligns more with Group A than not Group B and the second aligns more with Group B and not Group A. For example,
    * ("a photo of a car", "a photo of a car in the snow")
    * ("a photo of a dig", "a photo of a dog with black hair")
    * ("animals walking around", "animals drinking water during the day")
    * ("a photo of an object", "a drawing of an object")
    
    Again, I want to figure out the main differences between these two groups of images. List properties that holds more often for the images in group A compared to group B and vice versa. Your response:
    """

VQA_DIFF_PROMPT = """
    The following are the result of asking a VQA model {question} about the following two groups of captions:

    {text}

    Please write a list of hypotheses (separated by bullet points "-") of how images from
    Group A differ from those from Group B via their captions. Each hypothesis should be formatted as "Group A ... and Group B..." and should be with respect to the caption question. Here are three examples:
    - "Group A is cars in mostly outdoor environments and Group B is cars mostly indoor environments.”
    - "Group A is dogs with golden hair and Group B is cats with black hair."
    - "Group A is various animals walking around at night and Group B is various animals drinking water around during the day."
    - "Group A and Group B are similar."

    The answers should be pertaining to the content of the images, not the structure of the captions. Here are examples of incorrect answers:
    - "Group A has longer captions and Group B has shorter captions"
    - "Group A is more detailed and Group B is more general"
    
    Based on the two caption groups (A and B) from the above...
    """

RUIQI_DIFF_PROMPT = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to build an image classifier. Group A are the captions of the image about a class in the training set, and Group B are the test set. I want to figure out what kind of distribution shift are there.

    Come up with 10 short and distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "dog with a long tail"
    * "sunny"
    * "graphic art"
    * "bird with a brown beak"
    * "blurry"
    * "DSLR photo"
    * "person"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "people in various settings" corrected: "people"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images containing wooden elements" corrected: "wooden"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B, with each property being under 5 words. Your response:
"""


RUIQI_DIFF_PROMPT_LONGER_VICUNA = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"
    * incorrect: "More realistic images" corrected: "realistic images"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*").
    OUTPUT:
"""

RUIQI_DIFF_PROMPT_LONGER = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Also do not list more than one concept. Here are examples of bad outputs and their corrections:
    * incorrect: "various nature environments like lakes, forests, and mountains" corrected: "nature"
    * incorrect: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * incorrect: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * incorrect: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * incorrect: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"
    * incorrect: "More realistic images" corrected: "realistic images"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""

CLIP_FRIENDLY = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""


VLM_PROMPT = """
    This image contains two groups of images. 20 images from Group A are shown in the first two rows, while 20 images from Group B are shown in the last two rows.

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""

VLM_PROMPTv2 = """
    Here, we have two groups of medical chest X-ray images. Group A chest Xrays are shown in the first image, while Group B Chest Xrays are part of the second image. 

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 5 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "evidence of pleural effudion"
    * "hyper expanded but clear lung fields"
    * "Bibasilar atelectasis"
    * "acute cardiomegaly"
    * "mediastinal contours are normal"
    * "no acute cardiopulomonary abnormality"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. 
    Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various types of artifacts like tubes, wires, and clips" CORRECTED: "artifacts"
    * INCORRECT: "images of lungs with conditions (e.g. pneumonia, fibrosis)" CORRECTED: "lung conditions"
    * INCORRECT: "Presence of abnormal shadows" CORRECTED: "abnormal shadows"
    * INCORRECT: "Different types of opacities including nodular, reticular, and alveolar" CORRECTED: "opacities"
    * INCORRECT: "Images involving multiple pathologies" CORRECTED: "multiple pathologies"
    * INCORRECT: "More overexposed images" CORRECTED: "overexposed images"
    * INCORRECT: "Artifacts (surgical clips, monitoring devices)" CORRECTED: "artifacts"

      Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""




VLM_PROMPT_SEPERATE = """
    Here, we have two groups of medical chest X-ray images.The first image contains 20 images from Group A and the second image contains 20 images from Group B.

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""




CLIP_FRIENDLY_GROUP_A = """
    The following are the result of captioning a group of images:

    {text}

    I am a machine learning researcher trying to figure out the major commonalities within this group so I can better understand my data.

    Come up with 10 distinct concepts that appear often in the group. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "presence of ...", "images with ..." are incorrect. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vaccuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"

    Again, I want to figure out the common concepts in a group of images. List properties that hold most often for images (not captions) in the group. Answer with a list (separated by bullet points "*"). Your response:
"""

RUIQI_DIFF_PROMPT_MINIMAL_CONTEXT = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "captions about bird", or "caption with one word", or "detailed caption". Here are examples of bad outputs and their corrections:
    * bad output: "various nature environments like lakes, forests, and mountains" corrected: "nature environments"
    * bad output: "images of household object (e.g. bowl, vaccuum, lamp)" corrected: "household objects"
    * bad output: "Water-related scenes (ocean, river, catamaran)" corrected: "water" or "water-related"
    * bad output: "Different types of vehicles including cars, trucks, boats, and RVs" corrected: "vehicles"
    * bad output: "Images involving interaction between humans and animals" corrected: "interaction between humans and animals"

    Again, I want to figure out what kind of distribution shift are there. List properties that holds more often for the images in group A compared to group B. Your response:
"""

DIFFUSION_LLM_PROMPT = """
    The following are the result of captioning two groups of images generated by two different image generation models, with each pair of captions corresponding to the same generation prompt:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can correctly identify which model generated which image for unseen prompts.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*") . for example:
    * "dogs with brown hair"
    * "a cluttered scene"
    * "low quality"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption that can be fed into CLIP, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibiliites within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vaccuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images" 
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what the main differences are between these two image generation models so I can correctly identify which model generated which image. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""