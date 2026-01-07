
def create_vlm_prompt_report(num_differences, sample_reports):
      vlm_prompt_report = f"""
      List all hypothetical potential differences between sets of chest x-ray radiology scans.
      These could include but not limited to variations in tissue density, presence of abnormalities such as tumors,
      lesions, or fractures, and any noticeable changes in anatomical structures. 
      Give me exactly {num_differences} differences in the format of A vs B in a JSON file. 
      Store condition A and B in seperate fields in the JSON. The JSON format should be of the following:

       [
        {{ "condition_A": "*insert condition A*", "condition_B": "*insert condition B*" }},
          ...
       ]

      Ensure these distinctions reflect the detailed nuances characteristic of radiology reports. 
      They should not be broad classification differences but rather subtle, intricate variations. 
      Here are sample radiology reports to help you:

      {sample_reports}
      """
      return vlm_prompt_report




def create_prompt_dedup(differences):
      vlm_prompt_dedup =  f"""
            Below are hypothetical differences between chest X ray. For the below set of differences, remove any differences that are semantically and medically similar to each other. 
            Please be sure to tell me which differences were removed and explain your reasoning.

            {differences}

             Return the final differences, with duplicates removed, as a JSON in the following format:

     
            {{
            differences: [
            {{
                  "condition_A": "",
                  "condition_B": "",
            }},
            ...
             ]
            }}
      
            
      
      """

      return vlm_prompt_dedup

def create_classification_stage_prompt(difference, reports):
       prompt =  f"""We have the following condition of the format A vs B respectively: {difference}.
       Given the following {len(reports)} radiology reports, group each report into either having condition A or B or neither. 
       Classify each report into only one group exactly. Do not place a report in multiple groups.
       Provide reasoning and direct evidence in quotes from the report to justify each grouping. 
       Put the final output in a JSON with the following format:
      {{
      "group A": [
            {{
                  "report_index": "",
                  "reasoning": "",
                  "direct_evidence": "",
                 
            }},
            ...
      ],
      "group B": [
            {{
                  "report_index": "",
                  "reasoning": "",
                  "direct_evidence": "",
                  
            }},
            ...
      ],
      "neither": [
            {{
                  "report_index": "",
                  "reasoning": "",
                  "direct_evidence": "",
                  
            }},
            ...
      ]
      }}
      
      Please make sure to classify ALL the reports shown below:

      {reports}
      """

       return prompt




def create_prompt_detailed(num_differences):
      vlm_prompt_details = f"""
            List all hypothetical potential differences between sets of chest x ray radiology scans. 
            These could include but not limited to variations in tissue density, presence of abnormalities such as tumors, 
            lesions, or fractures, and any noticeable changes in anatomical structures. 
             Give me exactly {num_differences} differences in the format of A vs B in a JSON file. Store condition A and B in seperate fields, and store all the differences under key 'differences'
            Ensure these distinctions reflect the detailed nuances characteristic of radiology reports. 
            They should not be broad classification differences but rather subtle, intricate variations. 
      
      """

      return vlm_prompt_details