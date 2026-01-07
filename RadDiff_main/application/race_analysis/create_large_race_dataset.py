import pandas as pd
from pathlib import Path
import argparse
import json
import os
import numpy as np
from typing import List, Optional
from collections import defaultdict
from dotenv import load_dotenv


def find_image_path(dicom_id, subject_id, study_id, base_path=os.environ.get("base_path", "data")):
    """Generate the full path to an image file given DICOM ID and metadata"""
    p_dir = f"p{str(subject_id)[:2]}"
    image_path = Path(base_path) / "files" / p_dir / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    return str(image_path)


def map_ethnicity(e):
    """Map detailed ethnicity to race groups"""
    e = str(e).upper()

    if e.startswith("WHITE") or e == "PORTUGUESE":
        return "White"
    elif e.startswith("BLACK"):
        return "Black"
    elif e.startswith("ASIAN"):
        return "Asian"
    elif (e.startswith("HISPANIC") or e.startswith("LATINO") or e == "SOUTH AMERICAN"):
        return "Hispanic/Latino"
    elif e.startswith("AMERICAN INDIAN") or e.startswith("ALASKA NATIVE"):
        return "American Indian/Alaska Native"
    elif e.startswith("NATIVE HAWAIIAN") or "PACIFIC ISLANDER" in e:
        return "Native Hawaiian/Pacific Islander"
    elif "MULTIPLE RACE" in e:
        return "Multiple Race"
    else:
        return "Unknown/Declined"


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
        "pleural_effusion": {
            "icd9": ["511", "5110", "5111", "5118", "5119"],
            "icd10": ["J90", "J91", "J910", "J918"],
            "description": "Pleural Effusion"
        },
        "pulmonary_edema": {
            "icd9": ["5148"],
            "icd10": ["J81", "J810", "J811"],
            "description": "Pulmonary Edema"
        },
        "respiratory_failure": {
            "icd9": ["518", "5185", "51881", "51882", "51883", "51884"],
            "icd10": ["J96", "J960", "J9600", "J9601", "J9602", "J961", "J9610", "J9611", "J9612",
                      "J962", "J9620", "J9621", "J9622", "J969", "J9690", "J9691", "J9692"],
            "description": "Respiratory Failure"
        },
        "covid19": {
            "icd9": [],
            "icd10": ["U071", "U072", "J1289"],
            "description": "COVID-19"
        },
        "cardiomegaly": {
            "icd9": ["4293"],
            "icd10": ["I51", "I517"],
            "description": "Cardiomegaly/Enlarged Heart"
        }
    }
    return categories


def create_large_race_dataset(
    races_to_compare: List[str] = None,
    max_samples_per_race: Optional[int] = None,
    try_healthy_only: bool = True,
    filter_view_position: bool = False,
    allowed_views: List[str] = None,
    random_seed: int = 42
):
    """
    Create a LARGE dataset for training a race classifier on chest X-rays.

    Parameters:
    -----------
    races_to_compare : List[str], optional
        List of races to include. Default: ['White', 'Black', 'Asian']

    max_samples_per_race : int, optional
        Maximum samples per race. If None, uses all available data.

    try_healthy_only : bool
        If True, first attempts to create dataset from healthy patients only (no disease ICD codes).
        Falls back to mixed disease pool if insufficient healthy samples.

    filter_view_position : bool
        If True, only include PA (posterior-anterior) and AP (anterior-posterior) views.
        Default: False (include all views for maximum dataset size)

    allowed_views : List[str], optional
        List of allowed view positions. Default: ['PA', 'AP'] if filter_view_position=True

    random_seed : int
        Random seed for reproducibility
    """

    print("="*80)
    print("LARGE-SCALE RACE CLASSIFICATION DATASET CREATOR")
    print("="*80)
    print("\nPurpose: Create huge dataset for race classification ViT training")
    print()

    # Set defaults
    if races_to_compare is None:
        races_to_compare = ["White", "Black", "Asian"]
    if allowed_views is None and filter_view_position:
        allowed_views = ["PA", "AP"]

    np.random.seed(random_seed)

    # Load data
    print("[1/6] Loading MIMIC-CXR and MIMIC-IV data...")
    cxr = pd.read_csv(os.environ.get("cxr_path", "data/mimic-cxr-2.0.0-metadata.csv.gz"))
    adm = pd.read_csv(os.environ.get("adm_path", "data/admissions.csv.gz"))
    patients = pd.read_csv(os.environ.get("patients_path", "data/patients.csv.gz"))
    diagnoses = pd.read_csv(os.environ.get("diagnoses_path", "data/diagnoses_icd.csv.gz"))

    print(f"  - CXR images: {len(cxr):,}")
    print(f"  - Admissions: {len(adm):,}")
    print(f"  - Patients: {len(patients):,}")
    print(f"  - Diagnoses: {len(diagnoses):,}")

    # Map race
    print("\n[2/6] Mapping race categories...")
    adm["race_group"] = adm["race"].map(map_ethnicity)

    # Handle inconsistent race assignments - patients must have consistent race across admissions
    subject_races = adm.groupby('subject_id')['race_group'].agg(['nunique', lambda x: list(x.unique())])
    subject_races.columns = ['num_unique_races', 'race_list']
    inconsistent_subjects = subject_races[subject_races['num_unique_races'] > 1]

    if len(inconsistent_subjects) > 0:
        print(f"  - Excluding {len(inconsistent_subjects):,} subjects with inconsistent race assignments")
        adm = adm[~adm['subject_id'].isin(inconsistent_subjects.index)]

    # Get consistent race mapping per subject
    subject_race_map = adm.groupby('subject_id')['race_group'].first()

    # Merge CXR with demographics
    print("\n[3/6] Merging CXR data with demographics...")
    cxr_with_demo = cxr.merge(
        patients[["subject_id", "gender"]],
        on="subject_id",
        how="left"
    )

    # Add race from subject mapping
    cxr_with_demo["race_group"] = cxr_with_demo["subject_id"].map(subject_race_map)

    # Filter to specified races
    initial_count = len(cxr_with_demo)
    cxr_with_demo = cxr_with_demo[cxr_with_demo["race_group"].isin(races_to_compare)]
    print(f"  - Filtered to {races_to_compare}: {len(cxr_with_demo):,} images (removed {initial_count - len(cxr_with_demo):,})")

    # Filter by view position if requested
    if filter_view_position:
        initial_count = len(cxr_with_demo)
        cxr_with_demo = cxr_with_demo[cxr_with_demo["ViewPosition"].isin(allowed_views)]
        print(f"  - Filtered to {allowed_views} views: {len(cxr_with_demo):,} images (removed {initial_count - len(cxr_with_demo):,})")

    # Try healthy-only approach if requested
    final_dataset = None
    dataset_type = "mixed"

    if try_healthy_only:
        print("\n[4/6] Attempting to create HEALTHY-ONLY dataset...")

        # Get all subjects with ANY disease diagnosis
        disease_defs = define_disease_categories()
        all_disease_codes_icd9 = []
        all_disease_codes_icd10 = []
        for disease_info in disease_defs.values():
            all_disease_codes_icd9.extend(disease_info["icd9"])
            all_disease_codes_icd10.extend(disease_info["icd10"])

        diseased_hadm_ids = diagnoses[
            ((diagnoses["icd_version"] == 9) & (diagnoses["icd_code"].isin(all_disease_codes_icd9))) |
            ((diagnoses["icd_version"] == 10) & (diagnoses["icd_code"].isin(all_disease_codes_icd10)))
        ]["hadm_id"].unique()

        print(f"  - Found {len(diseased_hadm_ids):,} hospital admissions with tracked diseases")

        # Get subjects from diseased admissions
        diseased_subjects = adm[adm["hadm_id"].isin(diseased_hadm_ids)]["subject_id"].unique()
        print(f"  - {len(diseased_subjects):,} unique patients have disease diagnoses")

        # Filter to healthy subjects (no disease admissions)
        healthy_cxr = cxr_with_demo[~cxr_with_demo["subject_id"].isin(diseased_subjects)]

        print(f"  - Healthy chest X-rays available: {len(healthy_cxr):,}")

        # Check race distribution in healthy subset
        healthy_race_dist = healthy_cxr["race_group"].value_counts()
        print(f"\n  Healthy subset race distribution:")
        for race, count in healthy_race_dist.items():
            print(f"    - {race}: {count:,} images")

        # Check if we have sufficient healthy samples
        min_healthy = healthy_race_dist.min() if len(healthy_race_dist) > 0 else 0
        min_threshold = 1000  # Minimum samples per race to use healthy-only

        if min_healthy >= min_threshold:
            print(f"\n  ✓ Sufficient healthy samples (min={min_healthy:,} >= {min_threshold})")
            print(f"  → Using HEALTHY-ONLY dataset")
            final_dataset = healthy_cxr
            dataset_type = "healthy"
        else:
            print(f"\n  ✗ Insufficient healthy samples (min={min_healthy:,} < {min_threshold})")
            print(f"  → Falling back to MIXED DISEASE dataset")
    else:
        print("\n[4/6] Skipping healthy-only attempt (try_healthy_only=False)")

    # Use mixed disease approach if healthy-only didn't work or wasn't requested
    if final_dataset is None:
        print("\n[5/6] Creating MIXED DISEASE dataset...")
        print("  Strategy: Pool ALL chest X-rays regardless of disease status")
        print("  Rationale: Maximize dataset size for robust race classification")
        final_dataset = cxr_with_demo
        dataset_type = "mixed"
    else:
        print("\n[5/6] Using healthy-only dataset (skipping disease mixing)")

    # Sample data if max_samples_per_race specified
    if max_samples_per_race is not None:
        print(f"\n  Sampling up to {max_samples_per_race:,} images per race...")
        sampled_data = []
        for race in races_to_compare:
            race_data = final_dataset[final_dataset["race_group"] == race]
            n_samples = min(len(race_data), max_samples_per_race)
            sampled = race_data.sample(n=n_samples, random_state=random_seed)
            sampled_data.append(sampled)
            print(f"    - {race}: {n_samples:,} images")
        final_dataset = pd.concat(sampled_data, ignore_index=True)

    # Remove duplicates
    initial_count = len(final_dataset)
    final_dataset = final_dataset.drop_duplicates(subset=["dicom_id"])
    if initial_count != len(final_dataset):
        print(f"  - Removed {initial_count - len(final_dataset):,} duplicate images")

    # Generate image paths
    final_dataset["path"] = final_dataset.apply(
        lambda row: find_image_path(row["dicom_id"], row["subject_id"], row["study_id"]),
        axis=1
    )

    # Print final statistics
    print("\n[6/6] FINAL DATASET STATISTICS")
    print("="*80)
    print(f"\nDataset type: {dataset_type.upper()}")
    print(f"Total images: {len(final_dataset):,}")
    print(f"\nRace distribution:")

    race_stats = {}
    for race in races_to_compare:
        race_data = final_dataset[final_dataset["race_group"] == race]
        count = len(race_data)
        pct = 100 * count / len(final_dataset)
        race_stats[race] = count
        print(f"  - {race}: {count:,} images ({pct:.1f}%)")

    # Gender distribution
    print(f"\nGender distribution:")
    for race in races_to_compare:
        race_data = final_dataset[final_dataset["race_group"] == race]
        gender_dist = race_data["gender"].value_counts()
        print(f"  - {race}: {dict(gender_dist)}")

    # View distribution
    if "ViewPosition" in final_dataset.columns:
        print(f"\nView position distribution:")
        view_dist = final_dataset["ViewPosition"].value_counts()
        for view, count in view_dist.items():
            pct = 100 * count / len(final_dataset)
            print(f"  - {view}: {count:,} ({pct:.1f}%)")

    # Save outputs
    output_dir = Path("large_race_dataset")
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("SAVING DATASETS")
    print(f"{'='*80}")

    # Save full dataset CSV
    full_csv = output_dir / f"race_classification_{dataset_type}_full.csv"
    final_dataset[["dicom_id", "subject_id", "study_id", "race_group", "gender", "ViewPosition", "path"]].to_csv(
        full_csv, index=False
    )
    print(f"\n✓ Saved full dataset: {full_csv}")
    print(f"  Total: {len(final_dataset):,} images")

    # Save individual race CSVs
    print(f"\n✓ Saved individual race datasets:")
    base_path = os.environ.get("base_path", "data")

    for race in races_to_compare:
        race_data = final_dataset[final_dataset["race_group"] == race]
        safe_race = race.replace("/", "_").replace(" ", "_")

        # CSV format
        race_csv = output_dir / f"{dataset_type}_{safe_race}.csv"
        race_data[["dicom_id", "subject_id", "study_id", "race_group", "gender", "path"]].to_csv(
            race_csv, index=False
        )
        print(f"  - {race_csv.name}: {len(race_data):,} images")

    # Save training/validation splits (80/20)
    print(f"\n✓ Creating train/val splits (80/20)...")
    train_data = []
    val_data = []

    for race in races_to_compare:
        race_data = final_dataset[final_dataset["race_group"] == race].sample(frac=1, random_state=random_seed)
        n_train = int(0.8 * len(race_data))
        train_data.append(race_data.iloc[:n_train])
        val_data.append(race_data.iloc[n_train:])

    train_df = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=random_seed)
    val_df = pd.concat(val_data, ignore_index=True).sample(frac=1, random_state=random_seed)

    # Save train/val CSVs
    train_csv = output_dir / f"race_classification_{dataset_type}_train.csv"
    val_csv = output_dir / f"race_classification_{dataset_type}_val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"  - Training set: {len(train_df):,} images → {train_csv.name}")
    print(f"  - Validation set: {len(val_df):,} images → {val_csv.name}")

    # Create ViT-ready format (class labels as integers)
    print(f"\n✓ Creating ViT-ready format...")
    race_to_label = {race: i for i, race in enumerate(races_to_compare)}

    train_df["label"] = train_df["race_group"].map(race_to_label)
    val_df["label"] = val_df["race_group"].map(race_to_label)

    vit_train_csv = output_dir / f"vit_train_{dataset_type}.csv"
    vit_val_csv = output_dir / f"vit_val_{dataset_type}.csv"

    train_df[["path", "label", "race_group"]].to_csv(vit_train_csv, index=False)
    val_df[["path", "label", "race_group"]].to_csv(vit_val_csv, index=False)

    print(f"  - {vit_train_csv.name}: path, label (0-{len(races_to_compare)-1}), race_group")
    print(f"  - {vit_val_csv.name}: path, label (0-{len(races_to_compare)-1}), race_group")
    print(f"\n  Label mapping: {race_to_label}")

    # Save metadata
    metadata = {
        "dataset_type": dataset_type,
        "races": races_to_compare,
        "race_to_label": race_to_label,
        "total_images": len(final_dataset),
        "race_distribution": race_stats,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "filters": {
            "view_positions": allowed_views if filter_view_position else "all",
            "max_samples_per_race": max_samples_per_race,
        },
        "random_seed": random_seed
    }

    metadata_file = output_dir / f"metadata_{dataset_type}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata: {metadata_file}")

    # Print next steps
    print(f"\n{'='*80}")
    print("NEXT STEPS: ViT Training & Confidence-Based Filtering")

    return final_dataset, metadata


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create large-scale dataset for race classification in chest X-rays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # mixed disease approach
  python create_large_race_dataset.py --no-healthy-only

        """
    )

    parser.add_argument(
        "--races",
        nargs="+",
        default=None,
        help="Races to include (default: White Black Asian)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per race (default: use all available)"
    )

    parser.add_argument(
        "--no-healthy-only",
        action="store_false",
        dest="try_healthy_only",
        help="Skip trying healthy-only dataset, go straight to mixed disease"
    )

    parser.add_argument(
        "--filter-views",
        action="store_true",
        dest="filter_view_position",
        default=False,
        help="Filter to only PA/AP views (default: include all views)"
    )

    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Allowed view positions (default: PA AP)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    create_large_race_dataset(
        races_to_compare=args.races,
        max_samples_per_race=args.max_samples,
        try_healthy_only=args.try_healthy_only,
        filter_view_position=args.filter_view_position,
        allowed_views=args.views,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
