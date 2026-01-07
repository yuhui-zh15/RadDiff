import pandas as pd
from pathlib import Path
import argparse
import json
import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv


def find_image_path(dicom_id, subject_id, study_id, base_path=os.environ.get("base_path", "data")):
    """Generate the full path to an image file given DICOM ID and metadata"""
    p_dir = f"p{str(subject_id)[:2]}"
    image_path = Path(base_path) / "files" / p_dir / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    return str(image_path)


def create_covid_datasets(
    comparisons: List[str] = None,
    min_samples_per_group: int = 100,
    max_samples_per_group: int = 1000,
    match_demographics: bool = True,
    young_age_cutoff: int = 50,
    old_age_cutoff: int = 65,
    very_young_cutoff: int = 40,
    middle_age_min: int = 41,
    middle_age_max: int = 60
):
    """
    Create COVID-19 comparison datasets for RadDiff.

    Parameters:
    -----------
    comparisons : List[str], optional
        Which comparisons to create. Options:
        - "young_vs_old_covid": Young COVID patients vs old COVID patients
        - "covid_complications_vs_covid_only": COVID with comorbidities vs COVID without
        - "very_young_vs_middle_vs_old_covid": Three-way age comparison showing mortality gradient
        Note: Removed "covid_vs_no_covid" - MIMIC-IV has no healthy controls

    min_samples_per_group : int
        Minimum number of samples required per group

    max_samples_per_group : int
        Maximum number of samples per group

    match_demographics : bool
        Whether to match age/gender distributions between groups

    young_age_cutoff : int
        Maximum age for "young" group (default: 50)

    old_age_cutoff : int
        Minimum age for "old" group (default: 65)

    very_young_cutoff : int
        Maximum age for "very young" group in three-way comparison (default: 40)

    middle_age_min : int
        Minimum age for "middle" group in three-way comparison (default: 41)

    middle_age_max : int
        Maximum age for "middle" group in three-way comparison (default: 60)
    """

    print("="*80)
    print("COVID-19 COMPARISON DATASET CREATOR")
    print("="*80)

    # Set defaults
    if comparisons is None:
        comparisons = ["young_vs_old_covid", "covid_complications_vs_covid_only"]

    # Load data
    print("\n[1/5] Loading MIMIC-CXR and MIMIC-IV data...")
    cxr = pd.read_csv(os.environ.get("cxr_path", "data/mimic-cxr-2.0.0-metadata.csv.gz"))
    adm = pd.read_csv(os.environ.get("adm_path", "data/admissions.csv.gz"))
    patients = pd.read_csv(os.environ.get("patients_path", "data/patients.csv.gz"))
    diagnoses = pd.read_csv(os.environ.get("diagnoses_path", "data/diagnoses_icd.csv.gz"))

    print(f"  - CXR images: {len(cxr):,}")
    print(f"  - Admissions: {len(adm):,}")
    print(f"  - Patients: {len(patients):,}")
    print(f"  - Diagnoses: {len(diagnoses):,}")

    # Merge patient demographics
    print("\n[2/5] Processing COVID cases...")
    adm_with_demo = adm.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")

    # Find COVID cases
    covid_codes = ["U071", "U072"]  # U071 = COVID-19, U072 = COVID-19 virus not identified
    covid_dx = diagnoses[diagnoses["icd_code"].isin(covid_codes)]
    covid_hadm_ids = covid_dx["hadm_id"].unique()

    print(f"  - Found {len(covid_hadm_ids):,} COVID admissions")
    print(f"  - Unique COVID patients: {covid_dx['subject_id'].nunique():,}")

    # Mark COVID status
    adm_with_demo["is_covid"] = adm_with_demo["hadm_id"].isin(covid_hadm_ids)

    # Get diagnosis counts per admission (for complication analysis)
    dx_per_admission = diagnoses.groupby("hadm_id").size().to_dict()
    adm_with_demo["num_diagnoses"] = adm_with_demo["hadm_id"].map(dx_per_admission).fillna(0)

    # Merge with CXR
    merged = cxr.merge(adm_with_demo, on="subject_id", how="inner")

    # Statistics
    covid_cxr = merged[merged["is_covid"]]
    non_covid_cxr = merged[~merged["is_covid"]]

    print(f"\n  COVID patients:")
    print(f"    - {len(covid_cxr):,} X-ray images")
    print(f"    - {covid_cxr['subject_id'].nunique():,} unique patients")
    print(f"    - Mean age: {covid_cxr['anchor_age'].mean():.1f}")
    print(f"    - Mortality: {covid_cxr['hospital_expire_flag'].mean()*100:.1f}%")

    print(f"\n  Non-COVID patients:")
    print(f"    - {len(non_covid_cxr):,} X-ray images")
    print(f"    - {non_covid_cxr['subject_id'].nunique():,} unique patients")

    # Create output directory
    print("\n[3/5] Creating comparison datasets...")
    output_dir = Path("covid_datasets")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    # Three-way age comparison (Very Young vs Middle vs Old COVID)
    if "very_young_vs_middle_vs_old_covid" in comparisons:
        print(f"\n  {'='*70}")
        print(f"  COMPARISON: Very Young vs Middle Age vs Old COVID Patients")
        print(f"  {'='*70}")

        covid_data = merged[merged["is_covid"]].copy()

        very_young = covid_data[covid_data["anchor_age"] <= very_young_cutoff].copy()
        middle = covid_data[(covid_data["anchor_age"] >= middle_age_min) &
                           (covid_data["anchor_age"] <= middle_age_max)].copy()
        old = covid_data[covid_data["anchor_age"] > middle_age_max].copy()

        print(f"    Very Young (≤{very_young_cutoff}): {len(very_young):,} images, {very_young['subject_id'].nunique()} patients")
        print(f"      - Mortality: {very_young['hospital_expire_flag'].mean()*100:.1f}%")
        print(f"    Middle Age ({middle_age_min}-{middle_age_max}): {len(middle):,} images, {middle['subject_id'].nunique()} patients")
        print(f"      - Mortality: {middle['hospital_expire_flag'].mean()*100:.1f}%")
        print(f"    Old (>{middle_age_max}): {len(old):,} images, {old['subject_id'].nunique()} patients")
        print(f"      - Mortality: {old['hospital_expire_flag'].mean()*100:.1f}%")

        if len(very_young) < min_samples_per_group or len(middle) < min_samples_per_group or len(old) < min_samples_per_group:
            print(f"    ⚠️  Insufficient samples in one or more groups. Skipping.")
        else:
            # Match by gender across all three groups
            if match_demographics:
                print(f"\n    Matching by gender across three age groups...")
                matched_samples = []

                for gender in ["M", "F"]:
                    vy_g = very_young[very_young["gender"] == gender]
                    mid_g = middle[middle["gender"] == gender]
                    old_g = old[old["gender"] == gender]

                    # Sample the minimum number across all three groups
                    n_samples = min(len(vy_g), len(mid_g), len(old_g))

                    if n_samples > 0:
                        matched_samples.append(vy_g.sample(n=n_samples, random_state=42))
                        matched_samples.append(mid_g.sample(n=n_samples, random_state=42))
                        matched_samples.append(old_g.sample(n=n_samples, random_state=42))

                matched = pd.concat(matched_samples, ignore_index=True)
            else:
                n_vy = min(len(very_young), max_samples_per_group)
                n_mid = min(len(middle), max_samples_per_group)
                n_old = min(len(old), max_samples_per_group)

                matched = pd.concat([
                    very_young.sample(n=n_vy, random_state=42),
                    middle.sample(n=n_mid, random_state=42),
                    old.sample(n=n_old, random_state=42)
                ], ignore_index=True)

            # Remove duplicates
            matched = matched.drop_duplicates(subset=["dicom_id"])

            # Generate paths
            matched["path"] = matched.apply(
                lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
                axis=1
            )

            # Split into three groups
            vy_final = matched[matched["anchor_age"] <= very_young_cutoff]
            mid_final = matched[(matched["anchor_age"] >= middle_age_min) &
                               (matched["anchor_age"] <= middle_age_max)]
            old_final = matched[matched["anchor_age"] > middle_age_max]

            print(f"\n    Final cohort (gender-matched):")
            print(f"      - Very Young: {len(vy_final):,} images (age: {vy_final['anchor_age'].mean():.1f}±{vy_final['anchor_age'].std():.1f}, mortality: {vy_final['hospital_expire_flag'].mean()*100:.1f}%)")
            print(f"      - Middle Age: {len(mid_final):,} images (age: {mid_final['anchor_age'].mean():.1f}±{mid_final['anchor_age'].std():.1f}, mortality: {mid_final['hospital_expire_flag'].mean()*100:.1f}%)")
            print(f"      - Old: {len(old_final):,} images (age: {old_final['anchor_age'].mean():.1f}±{old_final['anchor_age'].std():.1f}, mortality: {old_final['hospital_expire_flag'].mean()*100:.1f}%)")

            # Create three pairwise comparisons for RadDiff (which expects two groups)
            # 1. Very Young vs Middle Age
            save_comparison_dataset(vy_final, mid_final, "very_young_covid", "middle_age_covid",
                                   "very_young_vs_middle_covid", output_dir, match_demographics, all_results)

            # 2. Middle Age vs Old
            save_comparison_dataset(mid_final, old_final, "middle_age_covid", "old_covid",
                                   "middle_vs_old_covid", output_dir, match_demographics, all_results)

            # 3. Very Young vs Old (extreme comparison)
            save_comparison_dataset(vy_final, old_final, "very_young_covid", "old_covid",
                                   "very_young_vs_old_covid", output_dir, match_demographics, all_results)

    # Print summary
    print("\n" + "="*80)
    print("[5/5] SUMMARY")
    print("="*80)
    print(f"\nCreated {len(all_results)} COVID comparison datasets:")
    for result in all_results:
        print(f"  - {result['comparison']}: {result['group1']} ({result['n1']}) vs {result['group2']} ({result['n2']})")

    print(f"\n✓ All datasets saved to: {output_dir}/")
    
    return all_results


def save_comparison_dataset(group1_data, group2_data, group1_label, group2_label,
                            comparison_name, output_dir, matched, all_results):
    """Helper function to save comparison datasets in CSV and JSONL format"""

    base_path = os.environ.get("base_path", "data")

    # Create CSV
    combined_data = pd.concat([
        pd.DataFrame({"group_name": group1_label, "path": group1_data["path"]}),
        pd.DataFrame({"group_name": group2_label, "path": group2_data["path"]})
    ], ignore_index=True)

    csv_file = output_dir / f"{comparison_name}.csv"
    combined_data.to_csv(csv_file, index=False)
    print(f"\n      Saved CSV: {len(combined_data):,} images to {csv_file.name}")

    # Create JSONL
    group1_relative_paths = [path.replace(base_path, "") for path in group1_data["path"]]
    group2_relative_paths = [path.replace(base_path, "") for path in group2_data["path"]]

    jsonl_comparison = {
        "set1": group1_label,
        "set2": group2_label,
        "set1_images": group1_relative_paths,
        "set2_images": group2_relative_paths,
        "comparison_type": comparison_name,
        "matched_demographics": matched
    }

    jsonl_file = output_dir / f"{comparison_name}.jsonl"
    with open(jsonl_file, 'w') as f:
        f.write(json.dumps(jsonl_comparison) + '\n')
    print(f"      Saved JSONL: {len(group1_relative_paths) + len(group2_relative_paths)} images to {jsonl_file.name}")

    # Store results
    all_results.append({
        "comparison": comparison_name,
        "group1": group1_label,
        "group2": group2_label,
        "n1": len(group1_data),
        "n2": len(group2_data)
    })


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create COVID-19 comparison datasets for RadDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  python create_covid_datasets.py --comparisons very_young_vs_middle_vs_old_covid

Available comparisons:
  - very_young_vs_middle_vs_old_covid: Three-way age comparison (creates 3 pairwise datasets)
        """
    )

    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=None,
        choices=["very_young_vs_middle_vs_old_covid"],
        help="Which comparisons to create"
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples per group (default: 100)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples per group (default: 1000)"
    )

    parser.add_argument(
        "--no-match-demographics",
        action="store_false",
        dest="match_demographics",
        help="Don't match demographics between groups"
    )

    parser.add_argument(
        "--young-age",
        type=int,
        default=50,
        help="Maximum age for 'young' group (default: 50)"
    )

    parser.add_argument(
        "--old-age",
        type=int,
        default=65,
        help="Minimum age for 'old' group (default: 65)"
    )

    parser.add_argument(
        "--very-young-age",
        type=int,
        default=40,
        help="Maximum age for 'very young' group in three-way comparison (default: 40)"
    )

    parser.add_argument(
        "--middle-age-min",
        type=int,
        default=41,
        help="Minimum age for 'middle' group in three-way comparison (default: 41)"
    )

    parser.add_argument(
        "--middle-age-max",
        type=int,
        default=60,
        help="Maximum age for 'middle' group in three-way comparison (default: 60)"
    )

    args = parser.parse_args()

    create_covid_datasets(
        comparisons=args.comparisons,
        min_samples_per_group=args.min_samples,
        max_samples_per_group=args.max_samples,
        match_demographics=args.match_demographics,
        young_age_cutoff=args.young_age,
        old_age_cutoff=args.old_age,
        very_young_cutoff=args.very_young_age,
        middle_age_min=args.middle_age_min,
        middle_age_max=args.middle_age_max
    )


if __name__ == "__main__":
    main()
