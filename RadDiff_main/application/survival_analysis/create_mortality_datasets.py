import pandas as pd
from pathlib import Path
import argparse
import json
import os
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv


def find_image_path(dicom_id, subject_id, study_id, base_path=os.environ.get("base_path", "data")):
    """Generate the full path to an image file given DICOM ID and metadata"""
    p_dir = f"p{str(subject_id)[:2]}"
    image_path = Path(base_path) / "files" / p_dir / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    return str(image_path)


def define_disease_categories():
    """Define common disease categories relevant to chest X-rays"""
    categories = {
        "pneumonia": {
            "icd9": ["486", "4800", "4801", "4802", "4803", "4808", "4809", "481", "482", "483", "485"],
            "icd10": ["J18", "J189", "J13", "J14", "J15", "J16", "J17", "J851"],
            "description": "Pneumonia (all types)"
        },
        "heart_failure": {
            "icd9": ["4280", "4281", "42820", "42821", "42822", "42823", "42830", "42831", "42832", "42833",
                     "42840", "42841", "42842", "42843", "4289"],
            "icd10": ["I50", "I500", "I501", "I502", "I503", "I504", "I509", "I5020", "I5021", "I5022",
                      "I5023", "I5030", "I5031", "I5032", "I5033", "I5040", "I5041", "I5042", "I5043"],
            "description": "Congestive Heart Failure"
        },
        "copd": {
            "icd9": ["490", "491", "4910", "4911", "4912", "49120", "49121", "49122", "4918", "4919",
                     "492", "4920", "4928", "496"],
            "icd10": ["J40", "J41", "J410", "J411", "J418", "J42", "J43", "J430", "J431", "J432", "J438",
                      "J439", "J44", "J440", "J441", "J449"],
            "description": "Chronic Obstructive Pulmonary Disease"
        },
        "respiratory_failure": {
            "icd9": ["518", "5185", "51881", "51882", "51883", "51884"],
            "icd10": ["J96", "J960", "J9600", "J9601", "J9602", "J961", "J9610", "J9611", "J9612",
                      "J962", "J9620", "J9621", "J9622", "J969", "J9690", "J9691", "J9692"],
            "description": "Respiratory Failure"
        },
        "sepsis": {
            "icd9": ["995", "99591", "99592", "78552"],
            "icd10": ["A41", "A419", "A4189", "R6520", "R6521"],
            "description": "Sepsis"
        },
        "ards": {
            "icd9": ["51882"],
            "icd10": ["J80"],
            "description": "Acute Respiratory Distress Syndrome"
        },
        "all": {
            "icd9": [],
            "icd10": [],
            "description": "All diseases (no filtering)"
        }
    }
    return categories


def create_severity_mortality_datasets(
    disease_categories: List[str] = None,
    min_samples_per_group: int = 100,
    max_samples_per_group: int = 1000,
    age_range: Tuple[int, int] = (18, 100),
    match_demographics: bool = True,
    long_survival_threshold: int = 365,  # Days to be considered "long survived"
):
    """
    Create mortality severity comparison datasets.

    Group A (Severity levels - mutually exclusive):
    - in_hospital_death: Died during hospitalization
    - 30_day_death: Died 1-30 days after discharge (NOT in-hospital)
    - 90_day_death: Died 31-90 days after admission (NOT in 30-day)

    Group B (Long survived):
    - long_survived: No death recorded OR survived > long_survival_threshold days

    Parameters:
    -----------
    disease_categories : List[str], optional
        List of disease categories to analyze. If None, uses ['pneumonia', 'heart_failure', 'copd'].

    min_samples_per_group : int
        Minimum number of samples required per outcome group

    max_samples_per_group : int
        Maximum number of samples per outcome group

    age_range : Tuple[int, int]
        Age range to include (min_age, max_age)

    match_demographics : bool
        Whether to match age and gender distributions between severity groups

    long_survival_threshold : int
        Number of days to be considered "long survived" (default: 365)
    """

    print("="*80)
    print("MORTALITY SEVERITY DATASET CREATOR")
    print("="*80)

    # Set defaults
    if disease_categories is None:
        disease_categories = ["pneumonia"]

    # Load data
    print("\n[1/7] Loading MIMIC-CXR and MIMIC-IV data...")
    cxr = pd.read_csv(os.environ.get("cxr_path", "data/mimic-cxr-2.0.0-metadata.csv.gz"))
    adm = pd.read_csv(os.environ.get("adm_path", "data/admissions.csv.gz"))
    patients = pd.read_csv(os.environ.get("patients_path", "data/patients.csv.gz"))
    diagnoses = pd.read_csv(os.environ.get("diagnoses_path", "data/diagnoses_icd.csv.gz"))

    print(f"  - CXR images: {len(cxr):,}")
    print(f"  - Admissions: {len(adm):,}")
    print(f"  - Patients: {len(patients):,}")
    print(f"  - Diagnoses: {len(diagnoses):,}")

    # Process mortality with severity levels
    print(f"\n[2/7] Processing mortality severity levels...")
    print(f"  Long survival threshold: >{long_survival_threshold} days")

    # Convert dates
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    patients["dod"] = pd.to_datetime(patients["dod"], errors='coerce')

    # Merge patient death dates
    adm_with_death = adm.merge(patients[["subject_id", "dod", "gender", "anchor_age"]], on="subject_id", how="left")

    # Calculate time to death from admission
    adm_with_death["days_to_death"] = (adm_with_death["dod"] - adm_with_death["admittime"]).dt.days

    # Define MUTUALLY EXCLUSIVE severity groups
    print("\n  Defining severity groups (mutually exclusive)...")

    # In-hospital death (most severe)
    adm_with_death["in_hospital_death"] = adm_with_death["hospital_expire_flag"] == 1

    # 30-day death (died within 30 days but NOT in hospital)
    adm_with_death["30_day_death"] = (
        (adm_with_death["days_to_death"] <= 30) &
        (adm_with_death["days_to_death"] >= 0) &
        (adm_with_death["hospital_expire_flag"] == 0)  # Exclude in-hospital deaths
    )

    # 90-day death (died 31-90 days after admission)
    adm_with_death["90_day_death"] = (
        (adm_with_death["days_to_death"] > 30) &
        (adm_with_death["days_to_death"] <= 90)
    )

    # Long survived (no death OR survived beyond threshold)
    adm_with_death["long_survived"] = (
        (adm_with_death["dod"].isna()) |
        (adm_with_death["days_to_death"] > long_survival_threshold)
    )

    # Print severity distribution
    print(f"\n  Severity distribution across all admissions:")
    print(f"    - In-hospital death: {adm_with_death['in_hospital_death'].sum():,} ({adm_with_death['in_hospital_death'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - 30-day death (post-discharge): {adm_with_death['30_day_death'].sum():,} ({adm_with_death['30_day_death'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - 90-day death (31-90 days): {adm_with_death['90_day_death'].sum():,} ({adm_with_death['90_day_death'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - Long survived (>{long_survival_threshold} days or alive): {adm_with_death['long_survived'].sum():,} ({adm_with_death['long_survived'].sum()/len(adm_with_death)*100:.2f}%)")

    # Filter age range
    print("\n[3/7] Filtering by age...")
    initial_count = len(adm_with_death)
    adm_with_death = adm_with_death[
        (adm_with_death["anchor_age"] >= age_range[0]) &
        (adm_with_death["anchor_age"] <= age_range[1])
    ]
    print(f"  - Filtered to age range {age_range[0]}-{age_range[1]}: {len(adm_with_death):,} admissions (removed {initial_count - len(adm_with_death):,})")

    # Get disease categories
    print("\n[4/7] Loading disease definitions...")
    disease_defs = define_disease_categories()

    # Process each disease category
    print("\n[5/7] Processing disease categories...")
    output_dir = Path("mortality_severity_datasets")
    output_dir.mkdir(exist_ok=True)

    all_results = []
    severity_groups = ["in_hospital_death", "30_day_death", "90_day_death"]

    for disease_name in disease_categories:
        if disease_name not in disease_defs:
            print(f"\n  ⚠️  Unknown disease category: {disease_name}")
            continue

        print(f"\n  {'='*70}")
        print(f"  Processing: {disease_defs[disease_name]['description']}")
        print(f"  {'='*70}")

        # Filter by disease (or take all if "all")
        if disease_name == "all":
            disease_adm = adm_with_death.copy()
            print(f"    Using all admissions: {len(disease_adm):,}")
        else:
            # Get ICD codes for this disease
            icd9_codes = disease_defs[disease_name]["icd9"]
            icd10_codes = disease_defs[disease_name]["icd10"]

            # Find admissions with this diagnosis
            disease_diagnoses = diagnoses[
                ((diagnoses["icd_version"] == 9) & (diagnoses["icd_code"].isin(icd9_codes))) |
                ((diagnoses["icd_version"] == 10) & (diagnoses["icd_code"].isin(icd10_codes)))
            ].copy()

            if len(disease_diagnoses) == 0:
                print(f"    ⚠️  No diagnoses found for {disease_name}")
                continue

            # Get unique admissions with this disease
            disease_hadm_ids = disease_diagnoses["hadm_id"].unique()
            print(f"    Found {len(disease_hadm_ids):,} admissions with {disease_name}")

            # Filter admissions to those with this disease
            disease_adm = adm_with_death[adm_with_death["hadm_id"].isin(disease_hadm_ids)].copy()

        # Merge with CXR images
        disease_merged = cxr.merge(disease_adm, on="subject_id", how="inner")
        print(f"    Matched to {len(disease_merged):,} chest X-ray images")

        # Show severity distribution for this disease
        print(f"\n    Severity distribution for {disease_name}:")
        for severity in severity_groups:
            count = disease_merged[severity].sum()
            print(f"      - {severity}: {count:,} images")
        long_count = disease_merged["long_survived"].sum()
        print(f"      - long_survived: {long_count:,} images")

        # Check if we have enough samples in long_survived group
        if long_count < min_samples_per_group:
            print(f"    ⚠️  Insufficient long survived samples (n={long_count}, need={min_samples_per_group}). Skipping.")
            continue

        # Create datasets for each severity vs long_survived
        safe_disease_name = disease_name.replace(" ", "_").replace("/", "_")

        for severity_level in severity_groups:
            severity_data = disease_merged[disease_merged[severity_level]].copy()
            survived_data = disease_merged[disease_merged["long_survived"]].copy()

            if len(severity_data) < min_samples_per_group:
                print(f"\n    ⚠️  Skipping {severity_level}: insufficient samples (n={len(severity_data)}, need={min_samples_per_group})")
                continue

            print(f"\n    Creating {severity_level} vs long_survived comparison...")

            # Match demographics if requested
            if match_demographics:
                print(f"      Matching demographics...")

                # Create age bins for matching
                disease_merged["age_bin"] = pd.cut(disease_merged["anchor_age"], bins=10, labels=False)

                matched_samples = []

                # Match within each age bin and gender
                for age_bin in disease_merged["age_bin"].unique():
                    if pd.isna(age_bin):
                        continue

                    for gender in ["M", "F"]:
                        age_gender_data = disease_merged[
                            (disease_merged["age_bin"] == age_bin) &
                            (disease_merged["gender"] == gender)
                        ].copy()

                        severity_subset = age_gender_data[age_gender_data[severity_level]].copy()
                        survived_subset = age_gender_data[age_gender_data["long_survived"]].copy()

                        # Sample equal numbers from each outcome
                        n_samples = min(len(severity_subset), len(survived_subset))

                        if n_samples > 0:
                            severity_sample = severity_subset.sample(n=n_samples, random_state=42)
                            survived_sample = survived_subset.sample(n=n_samples, random_state=42)
                            matched_samples.append(severity_sample)
                            matched_samples.append(survived_sample)

                if not matched_samples:
                    print(f"      ⚠️  Could not create matched samples for {severity_level}")
                    continue

                disease_matched = pd.concat(matched_samples, ignore_index=True)

            else:
                # No matching, just sample up to max
                n_severity = min(len(severity_data), max_samples_per_group)
                n_survived = min(len(survived_data), max_samples_per_group)

                disease_matched = pd.concat([
                    severity_data.sample(n=n_severity, random_state=42),
                    survived_data.sample(n=n_survived, random_state=42)
                ], ignore_index=True)

            # Remove duplicates
            disease_matched = disease_matched.drop_duplicates(subset=["dicom_id"])

            # Generate image paths
            disease_matched["path"] = disease_matched.apply(
                lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
                axis=1
            )

            # Get final groups
            severity_final = disease_matched[disease_matched[severity_level]]
            survived_final = disease_matched[disease_matched["long_survived"]]

            print(f"\n      Final cohort statistics:")
            print(f"        Total images: {len(disease_matched):,}")
            print(f"        - {severity_level}: {len(severity_final):,} images")
            print(f"        - long_survived: {len(survived_final):,} images")

            # Age statistics
            print(f"\n      Age distribution:")
            print(f"        - {severity_level}: mean={severity_final['anchor_age'].mean():.1f}, median={severity_final['anchor_age'].median():.1f}")
            print(f"        - long_survived: mean={survived_final['anchor_age'].mean():.1f}, median={survived_final['anchor_age'].median():.1f}")

            # Gender distribution
            print(f"\n      Gender distribution:")
            print(f"        - {severity_level}: {dict(severity_final['gender'].value_counts())}")
            print(f"        - long_survived: {dict(survived_final['gender'].value_counts())}")

            # Create labels
            severity_label = f"{severity_level}_{disease_name}"
            survived_label = f"long_survived_{disease_name}"

            # Create pairwise comparison dataset
            base_path = Path(os.environ.get("base_path", "data"))

            # Create CSV
            combined_data = pd.concat([
                pd.DataFrame({"group_name": severity_label, "path": severity_final["path"]}),
                pd.DataFrame({"group_name": survived_label, "path": survived_final["path"]})
            ], ignore_index=True)

            pair_file = output_dir / f"{safe_disease_name}_{severity_level}_vs_long_survived.csv"
            combined_data.to_csv(pair_file, index=False)
            print(f"        Saved CSV: {len(combined_data):,} images to {pair_file.name}")

            # Create JSONL
            severity_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in severity_final["path"]]
            survived_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in survived_final["path"]]

            jsonl_comparison = {
                "set1": severity_label,
                "set2": survived_label,
                "set1_images": severity_relative_paths,
                "set2_images": survived_relative_paths,
                "disease_category": disease_name,
                "severity_level": severity_level,
                "long_survival_threshold_days": long_survival_threshold,
                "matched_on": ["age", "gender"] if match_demographics else []
            }

            jsonl_file = output_dir / f"{safe_disease_name}_{severity_level}_vs_long_survived.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write(json.dumps(jsonl_comparison) + '\n')
            print(f"        Saved JSONL: {len(severity_relative_paths) + len(survived_relative_paths)} images to {jsonl_file.name}")

            # Store results for summary
            all_results.append({
                "disease": disease_name,
                "comparison": f"{severity_level} vs long_survived",
                "n_group1": len(severity_final),
                "n_group2": len(survived_final),
                "total": len(combined_data)
            })

        # Create severity-to-severity comparisons
        print(f"\n    Creating severity-to-severity comparisons...")

        severity_comparisons = [
            ("in_hospital_death", "30_day_death"),
            ("in_hospital_death", "90_day_death")
        ]

        for severity1, severity2 in severity_comparisons:
            severity1_data = disease_merged[disease_merged[severity1]].copy()
            severity2_data = disease_merged[disease_merged[severity2]].copy()

            if len(severity1_data) < min_samples_per_group or len(severity2_data) < min_samples_per_group:
                print(f"\n    ⚠️  Skipping {severity1} vs {severity2}: insufficient samples")
                continue

            print(f"\n    Creating {severity1} vs {severity2} comparison...")

            # Match demographics if requested
            if match_demographics:
                print(f"      Matching demographics...")

                # Create age bins for matching
                disease_merged["age_bin"] = pd.cut(disease_merged["anchor_age"], bins=10, labels=False)

                matched_samples = []

                # Match within each age bin and gender
                for age_bin in disease_merged["age_bin"].unique():
                    if pd.isna(age_bin):
                        continue

                    for gender in ["M", "F"]:
                        age_gender_data = disease_merged[
                            (disease_merged["age_bin"] == age_bin) &
                            (disease_merged["gender"] == gender)
                        ].copy()

                        sev1_subset = age_gender_data[age_gender_data[severity1]].copy()
                        sev2_subset = age_gender_data[age_gender_data[severity2]].copy()

                        # Sample equal numbers from each severity
                        n_samples = min(len(sev1_subset), len(sev2_subset))

                        if n_samples > 0:
                            sev1_sample = sev1_subset.sample(n=n_samples, random_state=42)
                            sev2_sample = sev2_subset.sample(n=n_samples, random_state=42)
                            matched_samples.append(sev1_sample)
                            matched_samples.append(sev2_sample)

                if not matched_samples:
                    print(f"      ⚠️  Could not create matched samples for {severity1} vs {severity2}")
                    continue

                severity_matched = pd.concat(matched_samples, ignore_index=True)

            else:
                # No matching, just sample up to max
                n_sev1 = min(len(severity1_data), max_samples_per_group)
                n_sev2 = min(len(severity2_data), max_samples_per_group)

                severity_matched = pd.concat([
                    severity1_data.sample(n=n_sev1, random_state=42),
                    severity2_data.sample(n=n_sev2, random_state=42)
                ], ignore_index=True)

            # Remove duplicates
            severity_matched = severity_matched.drop_duplicates(subset=["dicom_id"])

            # Generate image paths
            severity_matched["path"] = severity_matched.apply(
                lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
                axis=1
            )

            # Get final groups
            sev1_final = severity_matched[severity_matched[severity1]]
            sev2_final = severity_matched[severity_matched[severity2]]

            print(f"\n      Final cohort statistics:")
            print(f"        Total images: {len(severity_matched):,}")
            print(f"        - {severity1}: {len(sev1_final):,} images")
            print(f"        - {severity2}: {len(sev2_final):,} images")

            # Age statistics
            print(f"\n      Age distribution:")
            print(f"        - {severity1}: mean={sev1_final['anchor_age'].mean():.1f}, median={sev1_final['anchor_age'].median():.1f}")
            print(f"        - {severity2}: mean={sev2_final['anchor_age'].mean():.1f}, median={sev2_final['anchor_age'].median():.1f}")

            # Gender distribution
            print(f"\n      Gender distribution:")
            print(f"        - {severity1}: {dict(sev1_final['gender'].value_counts())}")
            print(f"        - {severity2}: {dict(sev2_final['gender'].value_counts())}")

            # Create labels
            sev1_label = f"{severity1}_{disease_name}"
            sev2_label = f"{severity2}_{disease_name}"

            # Create pairwise comparison dataset
            base_path = Path(os.environ.get("base_path", "data"))

            # Create CSV
            combined_data = pd.concat([
                pd.DataFrame({"group_name": sev1_label, "path": sev1_final["path"]}),
                pd.DataFrame({"group_name": sev2_label, "path": sev2_final["path"]})
            ], ignore_index=True)

            pair_file = output_dir / f"{safe_disease_name}_{severity1}_vs_{severity2}.csv"
            combined_data.to_csv(pair_file, index=False)
            print(f"        Saved CSV: {len(combined_data):,} images to {pair_file.name}")

            # Create JSONL
            sev1_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in sev1_final["path"]]
            sev2_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in sev2_final["path"]]

            jsonl_comparison = {
                "set1": sev1_label,
                "set2": sev2_label,
                "set1_images": sev1_relative_paths,
                "set2_images": sev2_relative_paths,
                "disease_category": disease_name,
                "comparison_type": "severity_to_severity",
                "severity1": severity1,
                "severity2": severity2,
                "matched_on": ["age", "gender"] if match_demographics else []
            }

            jsonl_file = output_dir / f"{safe_disease_name}_{severity1}_vs_{severity2}.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write(json.dumps(jsonl_comparison) + '\n')
            print(f"        Saved JSONL: {len(sev1_relative_paths) + len(sev2_relative_paths)} images to {jsonl_file.name}")

            # Store results for summary
            all_results.append({
                "disease": disease_name,
                "comparison": f"{severity1} vs {severity2}",
                "n_group1": len(sev1_final),
                "n_group2": len(sev2_final),
                "total": len(combined_data)
            })

    # Print summary
    print("\n" + "="*80)
    print("[6/7] SUMMARY")
    print("="*80)
    print(f"\nCreated {len(all_results)} severity-based mortality datasets:")
    for result in all_results:
        print(f"  - {result['disease']}: {result['comparison']} ({result['n_group1']} + {result['n_group2']} = {result['total']} images)")

    print(f"\n✓ All datasets saved to: {output_dir}/")
    print(f"\nSeverity groups (mutually exclusive):")
    print(f"  - in_hospital_death: Died during hospitalization")
    print(f"  - 30_day_death: Died 1-30 days after discharge (not in-hospital)")
    print(f"  - 90_day_death: Died 31-90 days after admission")
    print(f"  - long_survived: Survived >{long_survival_threshold} days or no recorded death")

    print(f"\nComparisons created:")
    print(f"  1. Each severity level vs long_survived (3 comparisons per disease)")
    print(f"  2. Severity-to-severity: in_hospital_death vs 30_day_death")
    print(f"  3. Severity-to-severity: in_hospital_death vs 90_day_death")

    print("\n[7/7] Next steps:")
    print("  Use these datasets with RadDiff to identify radiological features by severity")

    return all_results


def create_mortality_datasets(
    disease_categories: List[str] = None,
    min_samples_per_group: int = 100,
    max_samples_per_group: int = 1000,
    age_range: Tuple[int, int] = (18, 100),
    match_demographics: bool = True,
    time_window: str = "in_hospital",  # "in_hospital", "30_day", "90_day", "1_year"
    severity_bins: int = 3
):
    """
    Create mortality outcome comparison datasets (died vs survived).

    Parameters:
    -----------
    disease_categories : List[str], optional
        List of disease categories to analyze. If None, uses ['pneumonia', 'heart_failure', 'copd'].

    min_samples_per_group : int
        Minimum number of samples required per outcome group

    max_samples_per_group : int
        Maximum number of samples per outcome group

    age_range : Tuple[int, int]
        Age range to include (min_age, max_age)

    match_demographics : bool
        Whether to match age and gender distributions between died/survived groups

    time_window : str
        Mortality timeframe:
        - "in_hospital": Died during hospitalization
        - "30_day": Died within 30 days of admission
        - "90_day": Died within 90 days of admission
        - "1_year": Died within 1 year of admission

    severity_bins : int
        Number of severity bins to create for matching
    """

    print("="*80)
    print("MORTALITY OUTCOME DATASET CREATOR")
    print("="*80)

    # Set defaults
    if disease_categories is None:
        disease_categories = ["pneumonia"]

    # Load data
    print("\n[1/6] Loading MIMIC-CXR and MIMIC-IV data...")
    cxr = pd.read_csv(os.environ.get("cxr_path", "data/mimic-cxr-2.0.0-metadata.csv.gz"))
    adm = pd.read_csv(os.environ.get("adm_path", "data/admissions.csv.gz"))
    patients = pd.read_csv(os.environ.get("patients_path", "data/patients.csv.gz"))
    diagnoses = pd.read_csv(os.environ.get("diagnoses_path", "data/diagnoses_icd.csv.gz"))

    print(f"  - CXR images: {len(cxr):,}")
    print(f"  - Admissions: {len(adm):,}")
    print(f"  - Patients: {len(patients):,}")
    print(f"  - Diagnoses: {len(diagnoses):,}")

    # Process mortality
    print(f"\n[2/6] Processing mortality outcomes (window: {time_window})...")

    # Convert dates
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    patients["dod"] = pd.to_datetime(patients["dod"], errors='coerce')

    # Merge patient death dates
    adm_with_death = adm.merge(patients[["subject_id", "dod", "gender", "anchor_age"]], on="subject_id", how="left")

    # Calculate time to death
    adm_with_death["days_to_death"] = (adm_with_death["dod"] - adm_with_death["admittime"]).dt.days

    # Define mortality based on time window
    if time_window == "in_hospital":
        adm_with_death["died"] = adm_with_death["hospital_expire_flag"] == 1
        adm_with_death["survived"] = adm_with_death["hospital_expire_flag"] == 0
    elif time_window == "30_day":
        adm_with_death["died"] = (adm_with_death["days_to_death"] <= 30) & (adm_with_death["days_to_death"] >= 0)
        adm_with_death["survived"] = (adm_with_death["days_to_death"] > 30) | (adm_with_death["dod"].isna())
    elif time_window == "90_day":
        adm_with_death["died"] = (adm_with_death["days_to_death"] <= 90) & (adm_with_death["days_to_death"] >= 0)
        adm_with_death["survived"] = (adm_with_death["days_to_death"] > 90) | (adm_with_death["dod"].isna())
    elif time_window == "1_year":
        adm_with_death["died"] = (adm_with_death["days_to_death"] <= 365) & (adm_with_death["days_to_death"] >= 0)
        adm_with_death["survived"] = (adm_with_death["days_to_death"] > 365) | (adm_with_death["dod"].isna())

    mortality_rate = adm_with_death["died"].sum() / len(adm_with_death) * 100
    print(f"  - Mortality rate ({time_window}): {mortality_rate:.2f}%")
    print(f"  - Died: {adm_with_death['died'].sum():,} admissions")
    print(f"  - Survived: {adm_with_death['survived'].sum():,} admissions")

    # Filter age range
    print("\n[3/6] Filtering by age...")
    initial_count = len(adm_with_death)
    adm_with_death = adm_with_death[
        (adm_with_death["anchor_age"] >= age_range[0]) &
        (adm_with_death["anchor_age"] <= age_range[1])
    ]
    print(f"  - Filtered to age range {age_range[0]}-{age_range[1]}: {len(adm_with_death):,} admissions (removed {initial_count - len(adm_with_death):,})")

    # Get disease categories
    print("\n[4/6] Loading disease definitions...")
    disease_defs = define_disease_categories()

    # Process each disease category
    print("\n[5/6] Processing disease categories...")
    output_dir = Path("mortality_outcome_datasets")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for disease_name in disease_categories:
        if disease_name not in disease_defs:
            print(f"\n  ⚠️  Unknown disease category: {disease_name}")
            continue

        print(f"\n  {'='*70}")
        print(f"  Processing: {disease_defs[disease_name]['description']}")
        print(f"  {'='*70}")

        # Filter by disease (or take all if "all")
        if disease_name == "all":
            disease_adm = adm_with_death.copy()
            print(f"    Using all admissions: {len(disease_adm):,}")
        else:
            # Get ICD codes for this disease
            icd9_codes = disease_defs[disease_name]["icd9"]
            icd10_codes = disease_defs[disease_name]["icd10"]

            # Find admissions with this diagnosis
            disease_diagnoses = diagnoses[
                ((diagnoses["icd_version"] == 9) & (diagnoses["icd_code"].isin(icd9_codes))) |
                ((diagnoses["icd_version"] == 10) & (diagnoses["icd_code"].isin(icd10_codes)))
            ].copy()

            if len(disease_diagnoses) == 0:
                print(f"    ⚠️  No diagnoses found for {disease_name}")
                continue

            # Get unique admissions with this disease
            disease_hadm_ids = disease_diagnoses["hadm_id"].unique()
            print(f"    Found {len(disease_hadm_ids):,} admissions with {disease_name}")

            # Filter admissions to those with this disease
            disease_adm = adm_with_death[adm_with_death["hadm_id"].isin(disease_hadm_ids)].copy()

        # Merge with CXR images
        disease_merged = cxr.merge(disease_adm, on="subject_id", how="inner")
        print(f"    Matched to {len(disease_merged):,} chest X-ray images")

        # Show mortality distribution
        died_count = disease_merged["died"].sum()
        survived_count = disease_merged["survived"].sum()
        disease_mortality_rate = died_count / (died_count + survived_count) * 100
        print(f"\n    Mortality distribution for {disease_name}:")
        print(f"      - Died: {died_count:,} images ({disease_mortality_rate:.1f}%)")
        print(f"      - Survived: {survived_count:,} images ({100-disease_mortality_rate:.1f}%)")

        # Check if we have enough samples
        if min(died_count, survived_count) < min_samples_per_group:
            print(f"    ⚠️  Insufficient samples (min={min(died_count, survived_count)}, need={min_samples_per_group}). Skipping.")
            continue

        # If matching demographics, do propensity-like matching
        if match_demographics:
            print(f"\n    Matching demographics between died/survived groups...")

            # Create age bins for matching
            disease_merged["age_bin"] = pd.cut(disease_merged["anchor_age"], bins=10, labels=False)

            matched_samples = []

            # Match within each age bin and gender
            for age_bin in disease_merged["age_bin"].unique():
                if pd.isna(age_bin):
                    continue

                for gender in ["M", "F"]:
                    age_gender_data = disease_merged[
                        (disease_merged["age_bin"] == age_bin) &
                        (disease_merged["gender"] == gender)
                    ].copy()

                    died_data = age_gender_data[age_gender_data["died"]].copy()
                    survived_data = age_gender_data[age_gender_data["survived"]].copy()

                    # Sample equal numbers from each outcome
                    n_samples = min(len(died_data), len(survived_data))

                    if n_samples > 0:
                        died_sample = died_data.sample(n=n_samples, random_state=42)
                        survived_sample = survived_data.sample(n=n_samples, random_state=42)
                        matched_samples.append(died_sample)
                        matched_samples.append(survived_sample)

            if not matched_samples:
                print(f"    ⚠️  Could not create matched samples for {disease_name}")
                continue

            disease_matched = pd.concat(matched_samples, ignore_index=True)

        else:
            # No matching, just sample up to max
            died_data = disease_merged[disease_merged["died"]].copy()
            survived_data = disease_merged[disease_merged["survived"]].copy()

            n_died = min(len(died_data), max_samples_per_group)
            n_survived = min(len(survived_data), max_samples_per_group)

            disease_matched = pd.concat([
                died_data.sample(n=n_died, random_state=42),
                survived_data.sample(n=n_survived, random_state=42)
            ], ignore_index=True)

        # Remove duplicates
        disease_matched = disease_matched.drop_duplicates(subset=["dicom_id"])

        # Generate image paths
        disease_matched["path"] = disease_matched.apply(
            lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
            axis=1
        )

        # Print matching statistics
        died_final = disease_matched[disease_matched["died"]]
        survived_final = disease_matched[disease_matched["survived"]]

        print(f"\n    Final cohort statistics:")
        print(f"      Total images: {len(disease_matched):,}")
        print(f"      - Died: {len(died_final):,} images")
        print(f"      - Survived: {len(survived_final):,} images")

        # Age statistics
        print(f"\n    Age distribution:")
        print(f"      - Died: mean={died_final['anchor_age'].mean():.1f}, median={died_final['anchor_age'].median():.1f}, std={died_final['anchor_age'].std():.1f}")
        print(f"      - Survived: mean={survived_final['anchor_age'].mean():.1f}, median={survived_final['anchor_age'].median():.1f}, std={survived_final['anchor_age'].std():.1f}")

        # Gender distribution
        print(f"\n    Gender distribution:")
        print(f"      - Died: {dict(died_final['gender'].value_counts())}")
        print(f"      - Survived: {dict(survived_final['gender'].value_counts())}")

        # Create labels for CSV/JSONL
        safe_disease_name = disease_name.replace(" ", "_").replace("/", "_")
        died_label = f"died_{disease_name}"
        survived_label = f"survived_{disease_name}"

        # Save individual outcome datasets
        for outcome, label in [("died", died_label), ("survived", survived_label)]:
            outcome_data = disease_matched[disease_matched[outcome]].copy()
            outcome_file = output_dir / f"{safe_disease_name}_{outcome}.csv"
            outcome_csv = pd.DataFrame({
                "group_name": label,
                "path": outcome_data["path"]
            })
            outcome_csv.to_csv(outcome_file, index=False)

        # Create pairwise comparison dataset
        print(f"\n    Creating died vs survived comparison...")
        base_path = Path(os.environ.get("base_path", "data"))

        # Create CSV
        combined_data = pd.concat([
            pd.DataFrame({"group_name": died_label, "path": died_final["path"]}),
            pd.DataFrame({"group_name": survived_label, "path": survived_final["path"]})
        ], ignore_index=True)

        pair_file = output_dir / f"{safe_disease_name}_died_vs_survived.csv"
        combined_data.to_csv(pair_file, index=False)
        print(f"      Saved CSV: {len(combined_data):,} images to {pair_file.name}")

        # Create JSONL
        died_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in died_final["path"]]
        survived_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in survived_final["path"]]

        jsonl_comparison = {
            "set1": died_label,
            "set2": survived_label,
            "set1_images": died_relative_paths,
            "set2_images": survived_relative_paths,
            "disease_category": disease_name,
            "mortality_window": time_window,
            "matched_on": ["age", "gender"] if match_demographics else []
        }

        jsonl_file = output_dir / f"{safe_disease_name}_died_vs_survived.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps(jsonl_comparison) + '\n')
        print(f"      Saved JSONL: {len(died_relative_paths) + len(survived_relative_paths)} images to {jsonl_file.name}")

        # Store results for summary
        all_results.append({
            "disease": disease_name,
            "comparison": "Died vs Survived",
            "n_died": len(died_final),
            "n_survived": len(survived_final),
            "total": len(combined_data)
        })

    # Print summary
    print("\n" + "="*80)
    print("[6/6] SUMMARY")
    print("="*80)
    print(f"\nCreated {len(all_results)} mortality outcome datasets:")
    for result in all_results:
        print(f"  - {result['disease']}: {result['comparison']} ({result['n_died']} + {result['n_survived']} = {result['total']} images)")

    print(f"\n✓ All datasets saved to: {output_dir}/")
    print(f"\nMortality window: {time_window}")

    return all_results


def create_mortality_vs_hospital_death_datasets(
    disease_categories: List[str] = None,
    min_samples_per_group: int = 100,
    max_samples_per_group: int = 1000,
    age_range: Tuple[int, int] = (18, 100),
    match_demographics: bool = True,
    long_survival_threshold: int = 365,
):
    """
    Create datasets comparing different mortality timeframes vs in-hospital death.

    This creates comparisons of:
    - 30_day_death vs died_in_hospital
    - 90_day_death vs died_in_hospital
    - long_survived vs died_in_hospital

    Parameters are same as create_severity_mortality_datasets
    """

    print("="*80)
    print("MORTALITY VS HOSPITAL DEATH DATASET CREATOR")
    print("="*80)

    # Set defaults
    if disease_categories is None:
        disease_categories = ["pneumonia"]

    # Load data
    print("\n[1/7] Loading MIMIC-CXR and MIMIC-IV data...")
    cxr = pd.read_csv(os.environ.get("cxr_path", "data/mimic-cxr-2.0.0-metadata.csv.gz"))
    adm = pd.read_csv(os.environ.get("adm_path", "data/admissions.csv.gz"))
    patients = pd.read_csv(os.environ.get("patients_path", "data/patients.csv.gz"))
    diagnoses = pd.read_csv(os.environ.get("diagnoses_path", "data/diagnoses_icd.csv.gz"))

    print(f"  - CXR images: {len(cxr):,}")
    print(f"  - Admissions: {len(adm):,}")
    print(f"  - Patients: {len(patients):,}")
    print(f"  - Diagnoses: {len(diagnoses):,}")

    # Process mortality with severity levels
    print(f"\n[2/7] Processing mortality severity levels...")
    print(f"  Long survival threshold: >{long_survival_threshold} days")

    # Convert dates
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])
    patients["dod"] = pd.to_datetime(patients["dod"], errors='coerce')

    # Merge patient death dates
    adm_with_death = adm.merge(patients[["subject_id", "dod", "gender", "anchor_age"]], on="subject_id", how="left")

    # Calculate time to death from admission
    adm_with_death["days_to_death"] = (adm_with_death["dod"] - adm_with_death["admittime"]).dt.days

    # Define mortality groups
    print("\n  Defining mortality groups...")

    # In-hospital death
    adm_with_death["died_in_hospital"] = adm_with_death["hospital_expire_flag"] == 1

    # 30-day death (died within 30 days but NOT in hospital)
    adm_with_death["30_day_death"] = (
        (adm_with_death["days_to_death"] <= 30) &
        (adm_with_death["days_to_death"] >= 0) &
        (adm_with_death["hospital_expire_flag"] == 0)
    )

    # 90-day death (died 31-90 days after admission)
    adm_with_death["90_day_death"] = (
        (adm_with_death["days_to_death"] > 30) &
        (adm_with_death["days_to_death"] <= 90)
    )

    # Long survived (no death OR survived beyond threshold)
    adm_with_death["long_survived"] = (
        (adm_with_death["dod"].isna()) |
        (adm_with_death["days_to_death"] > long_survival_threshold)
    )

    # Print distribution
    print(f"\n  Mortality distribution across all admissions:")
    print(f"    - Died in hospital: {adm_with_death['died_in_hospital'].sum():,} ({adm_with_death['died_in_hospital'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - 30-day death (post-discharge): {adm_with_death['30_day_death'].sum():,} ({adm_with_death['30_day_death'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - 90-day death (31-90 days): {adm_with_death['90_day_death'].sum():,} ({adm_with_death['90_day_death'].sum()/len(adm_with_death)*100:.2f}%)")
    print(f"    - Long survived (>{long_survival_threshold} days or alive): {adm_with_death['long_survived'].sum():,} ({adm_with_death['long_survived'].sum()/len(adm_with_death)*100:.2f}%)")

    # Filter age range
    print("\n[3/7] Filtering by age...")
    initial_count = len(adm_with_death)
    adm_with_death = adm_with_death[
        (adm_with_death["anchor_age"] >= age_range[0]) &
        (adm_with_death["anchor_age"] <= age_range[1])
    ]
    print(f"  - Filtered to age range {age_range[0]}-{age_range[1]}: {len(adm_with_death):,} admissions (removed {initial_count - len(adm_with_death):,})")

    # Get disease categories
    print("\n[4/7] Loading disease definitions...")
    disease_defs = define_disease_categories()

    # Process each disease category
    print("\n[5/7] Processing disease categories...")
    output_dir = Path("mortality_vs_hospital_datasets")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    # Define the comparisons we want to create
    comparison_groups = [
        ("30_day_death", "died_in_hospital"),
        ("90_day_death", "died_in_hospital"),
        ("long_survived", "died_in_hospital")
    ]

    for disease_name in disease_categories:
        if disease_name not in disease_defs:
            print(f"\n  ⚠️  Unknown disease category: {disease_name}")
            continue

        print(f"\n  {'='*70}")
        print(f"  Processing: {disease_defs[disease_name]['description']}")
        print(f"  {'='*70}")

        # Filter by disease (or take all if "all")
        if disease_name == "all":
            disease_adm = adm_with_death.copy()
            print(f"    Using all admissions: {len(disease_adm):,}")
        else:
            # Get ICD codes for this disease
            icd9_codes = disease_defs[disease_name]["icd9"]
            icd10_codes = disease_defs[disease_name]["icd10"]

            # Find admissions with this diagnosis
            disease_diagnoses = diagnoses[
                ((diagnoses["icd_version"] == 9) & (diagnoses["icd_code"].isin(icd9_codes))) |
                ((diagnoses["icd_version"] == 10) & (diagnoses["icd_code"].isin(icd10_codes)))
            ].copy()

            if len(disease_diagnoses) == 0:
                print(f"    ⚠️  No diagnoses found for {disease_name}")
                continue

            # Get unique admissions with this disease
            disease_hadm_ids = disease_diagnoses["hadm_id"].unique()
            print(f"    Found {len(disease_hadm_ids):,} admissions with {disease_name}")

            # Filter admissions to those with this disease
            disease_adm = adm_with_death[adm_with_death["hadm_id"].isin(disease_hadm_ids)].copy()

        # Merge with CXR images
        disease_merged = cxr.merge(disease_adm, on="subject_id", how="inner")
        print(f"    Matched to {len(disease_merged):,} chest X-ray images")

        # Show distribution for this disease
        print(f"\n    Mortality distribution for {disease_name}:")
        print(f"      - Died in hospital: {disease_merged['died_in_hospital'].sum():,} images")
        print(f"      - 30-day death: {disease_merged['30_day_death'].sum():,} images")
        print(f"      - 90-day death: {disease_merged['90_day_death'].sum():,} images")
        print(f"      - Long survived: {disease_merged['long_survived'].sum():,} images")

        # Check if we have enough hospital death samples
        hospital_death_count = disease_merged["died_in_hospital"].sum()
        if hospital_death_count < min_samples_per_group:
            print(f"    ⚠️  Insufficient hospital death samples (n={hospital_death_count}, need={min_samples_per_group}). Skipping.")
            continue

        # Create datasets for each comparison
        safe_disease_name = disease_name.replace(" ", "_").replace("/", "_")

        for group1, group2 in comparison_groups:
            group1_data = disease_merged[disease_merged[group1]].copy()
            group2_data = disease_merged[disease_merged[group2]].copy()

            if len(group1_data) < min_samples_per_group or len(group2_data) < min_samples_per_group:
                print(f"\n    ⚠️  Skipping {group1} vs {group2}: insufficient samples (n1={len(group1_data)}, n2={len(group2_data)}, need={min_samples_per_group})")
                continue

            print(f"\n    Creating {group1} vs {group2} comparison...")

            # Match demographics if requested
            if match_demographics:
                print(f"      Matching demographics...")

                # Create age bins for matching
                disease_merged["age_bin"] = pd.cut(disease_merged["anchor_age"], bins=10, labels=False)

                matched_samples = []

                # Match within each age bin and gender
                for age_bin in disease_merged["age_bin"].unique():
                    if pd.isna(age_bin):
                        continue

                    for gender in ["M", "F"]:
                        age_gender_data = disease_merged[
                            (disease_merged["age_bin"] == age_bin) &
                            (disease_merged["gender"] == gender)
                        ].copy()

                        g1_subset = age_gender_data[age_gender_data[group1]].copy()
                        g2_subset = age_gender_data[age_gender_data[group2]].copy()

                        # Sample equal numbers from each group
                        n_samples = min(len(g1_subset), len(g2_subset))

                        if n_samples > 0:
                            g1_sample = g1_subset.sample(n=n_samples, random_state=42)
                            g2_sample = g2_subset.sample(n=n_samples, random_state=42)
                            matched_samples.append(g1_sample)
                            matched_samples.append(g2_sample)

                if not matched_samples:
                    print(f"      ⚠️  Could not create matched samples for {group1} vs {group2}")
                    continue

                comparison_matched = pd.concat(matched_samples, ignore_index=True)

            else:
                # No matching, just sample up to max
                n_g1 = min(len(group1_data), max_samples_per_group)
                n_g2 = min(len(group2_data), max_samples_per_group)

                comparison_matched = pd.concat([
                    group1_data.sample(n=n_g1, random_state=42),
                    group2_data.sample(n=n_g2, random_state=42)
                ], ignore_index=True)

            # Remove duplicates
            comparison_matched = comparison_matched.drop_duplicates(subset=["dicom_id"])

            # Generate image paths
            comparison_matched["path"] = comparison_matched.apply(
                lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
                axis=1
            )

            # Get final groups
            g1_final = comparison_matched[comparison_matched[group1]]
            g2_final = comparison_matched[comparison_matched[group2]]

            print(f"\n      Final cohort statistics:")
            print(f"        Total images: {len(comparison_matched):,}")
            print(f"        - {group1}: {len(g1_final):,} images")
            print(f"        - {group2}: {len(g2_final):,} images")

            # Age statistics
            print(f"\n      Age distribution:")
            print(f"        - {group1}: mean={g1_final['anchor_age'].mean():.1f}, median={g1_final['anchor_age'].median():.1f}")
            print(f"        - {group2}: mean={g2_final['anchor_age'].mean():.1f}, median={g2_final['anchor_age'].median():.1f}")

            # Gender distribution
            print(f"\n      Gender distribution:")
            print(f"        - {group1}: {dict(g1_final['gender'].value_counts())}")
            print(f"        - {group2}: {dict(g2_final['gender'].value_counts())}")

            # Create labels
            g1_label = f"{group1}_{disease_name}"
            g2_label = f"{group2}_{disease_name}"

            # Create pairwise comparison dataset
            base_path = Path(os.environ.get("base_path", "data"))

            # Create CSV
            combined_data = pd.concat([
                pd.DataFrame({"group_name": g1_label, "path": g1_final["path"]}),
                pd.DataFrame({"group_name": g2_label, "path": g2_final["path"]})
            ], ignore_index=True)

            pair_file = output_dir / f"{safe_disease_name}_{group1}_vs_{group2}.csv"
            combined_data.to_csv(pair_file, index=False)
            print(f"        Saved CSV: {len(combined_data):,} images to {pair_file.name}")

            # Create JSONL
            g1_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in g1_final["path"]]
            g2_relative_paths = [path.replace(os.environ.get("base_path", "data"), "") for path in g2_final["path"]]

            jsonl_comparison = {
                "set1": g1_label,
                "set2": g2_label,
                "set1_images": g1_relative_paths,
                "set2_images": g2_relative_paths,
                "disease_category": disease_name,
                "comparison_type": f"{group1}_vs_{group2}",
                "long_survival_threshold_days": long_survival_threshold,
                "matched_on": ["age", "gender"] if match_demographics else []
            }

            jsonl_file = output_dir / f"{safe_disease_name}_{group1}_vs_{group2}.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write(json.dumps(jsonl_comparison) + '\n')
            print(f"        Saved JSONL: {len(g1_relative_paths) + len(g2_relative_paths)} images to {jsonl_file.name}")

            # Store results for summary
            all_results.append({
                "disease": disease_name,
                "comparison": f"{group1} vs {group2}",
                "n_group1": len(g1_final),
                "n_group2": len(g2_final),
                "total": len(combined_data)
            })

    # Print summary
    print("\n" + "="*80)
    print("[6/7] SUMMARY")
    print("="*80)
    print(f"\nCreated {len(all_results)} mortality vs hospital death datasets:")
    for result in all_results:
        print(f"  - {result['disease']}: {result['comparison']} ({result['n_group1']} + {result['n_group2']} = {result['total']} images)")

    print(f"\n✓ All datasets saved to: {output_dir}/")
    print(f"\nComparisons created:")
    print(f"  1. 30_day_death vs died_in_hospital")
    print(f"  2. 90_day_death vs died_in_hospital")
    print(f"  3. long_survived vs died_in_hospital")

    print("\n[7/7] Next steps:")
    print("  Use these datasets with RadDiff to identify radiological differences")

    return all_results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create mortality outcome datasets for RadDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Compare mortality timeframes vs hospital death
  python create_mortality_datasets.py --mode vs-hospital --diseases pneumonia

  # Severity-based comparison 
  python create_mortality_datasets.py --mode severity --diseases pneumonia

  # Specific disease with 30-day mortality (binary mode)
  python create_mortality_datasets.py --mode binary --diseases pneumonia --time-window 30_day

  # Severity mode with custom survival threshold
  python create_mortality_datasets.py --mode severity --long-survival 730


Available disease categories:
  - pneumonia: Pneumonia (all types)
  - heart_failure: Congestive Heart Failure
  - copd: Chronic Obstructive Pulmonary Disease
  - respiratory_failure: Respiratory Failure
  - sepsis: Sepsis
  - ards: Acute Respiratory Distress Syndrome
  - all: All diseases (no filtering)

Modes:
  - vs-hospital: Compare different mortality timeframes vs in-hospital death
    * 30_day_death vs died_in_hospital
    * 90_day_death vs died_in_hospital
    * long_survived vs died_in_hospital
  - severity: Compare mutually exclusive severity levels vs long-term survivors
    * Group A: in_hospital_death, 30_day_death, 90_day_death (mutually exclusive)
    * Group B: long_survived (>365 days or no death)
  - binary: Traditional died vs survived comparison within a time window
    * Time windows: in_hospital, 30_day, 90_day, 1_year
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="severity",
        choices=["severity", "binary", "vs-hospital"],
        help="Dataset creation mode: 'severity' for gradient analysis, 'binary' for died vs survived, 'vs-hospital' for comparisons against hospital death (default: severity)"
    )

    parser.add_argument(
        "--diseases",
        nargs="+",
        default=None,
        help="Disease categories to analyze (default: pneumonia heart_failure copd)"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples per outcome group (default: 100)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples per outcome group (default: 1000)"
    )

    parser.add_argument(
        "--age-min",
        type=int,
        default=18,
        help="Minimum age (default: 18)"
    )

    parser.add_argument(
        "--age-max",
        type=int,
        default=100,
        help="Maximum age (default: 100)"
    )

    parser.add_argument(
        "--no-match-demographics",
        action="store_false",
        dest="match_demographics",
        help="Don't match age and gender between died/survived groups"
    )

    parser.add_argument(
        "--time-window",
        type=str,
        default="in_hospital",
        choices=["in_hospital", "30_day", "90_day", "1_year"],
        help="Mortality timeframe for binary mode (default: in_hospital)"
    )

    parser.add_argument(
        "--long-survival",
        type=int,
        default=365,
        help="Days threshold for long survival in severity mode (default: 365)"
    )

    args = parser.parse_args()

    if args.mode == "severity":
        create_severity_mortality_datasets(
            disease_categories=args.diseases,
            min_samples_per_group=args.min_samples,
            max_samples_per_group=args.max_samples,
            age_range=(args.age_min, args.age_max),
            match_demographics=args.match_demographics,
            long_survival_threshold=args.long_survival
        )
    elif args.mode == "vs-hospital":
        create_mortality_vs_hospital_death_datasets(
            disease_categories=args.diseases,
            min_samples_per_group=args.min_samples,
            max_samples_per_group=args.max_samples,
            age_range=(args.age_min, args.age_max),
            match_demographics=args.match_demographics,
            long_survival_threshold=args.long_survival
        )
    else:  # binary mode
        create_mortality_datasets(
            disease_categories=args.diseases,
            min_samples_per_group=args.min_samples,
            max_samples_per_group=args.max_samples,
            age_range=(args.age_min, args.age_max),
            match_demographics=args.match_demographics,
            time_window=args.time_window
        )


if __name__ == "__main__":
    main()
