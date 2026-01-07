"""
Train a Vision Transformer (ViT) to classify race from chest X-rays.

This is part of a bootstrap approach:
1. Train ViT on large mixed-disease dataset
2. Extract predictions with confidence scores
3. Filter high-confidence examples to identify images with strong race signals to models

The goal is to find chest X-rays that signal racial differences to trained models,
so we can identify features that models learn to distinguish races.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import argparse
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class ChestXRayRaceDataset(Dataset):
    """Dataset for chest X-rays with race labels"""

    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to CSV with columns: path, label, race_group
            transform: torchvision transforms to apply
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Validate that files exist (sample check)
        if len(self.data) > 0:
            sample_path = self.data.iloc[0]['path']
            if not Path(sample_path).exists():
                raise FileNotFoundError(f"Sample image not found: {sample_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['path']
        label = row['label']

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, image_path


def get_transforms(augment=True, image_size=224):
    """Get image transforms for training/validation"""

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    return transform


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds


def extract_predictions_with_confidence(model, dataloader, device, race_names):
    """
    Extract predictions with confidence scores for all images.
    This is the KEY step for bootstrap filtering.
    """
    model.eval()

    results = []

    pbar = tqdm(dataloader, desc='Extracting confidence scores')
    with torch.no_grad():
        for images, labels, paths in pbar:
            images = images.to(device)

            # Get logits and convert to probabilities
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions and confidence
            max_probs, predictions = probs.max(1)

            # Store results
            for i in range(len(images)):
                results.append({
                    'path': paths[i],
                    'true_label': labels[i].item(),
                    'true_race': race_names[labels[i].item()],
                    'pred_label': predictions[i].item(),
                    'pred_race': race_names[predictions[i].item()],
                    'confidence': max_probs[i].item(),
                    'prob_white': probs[i][0].item() if len(probs[i]) > 0 else 0,
                    'prob_black': probs[i][1].item() if len(probs[i]) > 1 else 0,
                    'prob_asian': probs[i][2].item() if len(probs[i]) > 2 else 0,
                })

    return pd.DataFrame(results)


def filter_high_confidence_examples(predictions_df, confidence_threshold=0.95, top_n_per_race=1000):
    """
    Filter high-confidence predictions to find images with strongest race signals.
    This is the bootstrap filtering step.
    """
    print(f"\n{'='*80}")
    print("BOOTSTRAP FILTERING: High-Confidence Race Examples")
    print(f"{'='*80}")

    # Filter by confidence threshold
    high_conf = predictions_df[predictions_df['confidence'] >= confidence_threshold]
    print(f"\nImages with confidence >= {confidence_threshold}: {len(high_conf):,}")

    # Get correct predictions only
    correct_high_conf = high_conf[high_conf['true_label'] == high_conf['pred_label']]
    print(f"Correct high-confidence predictions: {len(correct_high_conf):,}")

    # Get top N per race
    filtered_examples = {}
    for race in predictions_df['true_race'].unique():
        race_examples = correct_high_conf[correct_high_conf['true_race'] == race]
        top_examples = race_examples.nlargest(top_n_per_race, 'confidence')
        filtered_examples[race] = top_examples

        print(f"\n{race}:")
        print(f"  - High-confidence correct: {len(race_examples):,}")
        print(f"  - Top {top_n_per_race}: {len(top_examples):,}")
        if len(top_examples) > 0:
            print(f"  - Confidence range: {top_examples['confidence'].min():.4f} - {top_examples['confidence'].max():.4f}")
            print(f"  - Mean confidence: {top_examples['confidence'].mean():.4f}")

    return filtered_examples


def main():
    parser = argparse.ArgumentParser(
        description="Train ViT for race classification on chest X-rays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--data-dir', type=str, default='small_race_dataset',
                       help='Directory containing training data')
    parser.add_argument('--dataset-type', type=str, default='mixed',
                       choices=['mixed', 'healthy'],
                       help='Dataset type to use')
    parser.add_argument('--model', type=str, default='deit_small_patch16_224',
                       help='ViT model architecture')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size for ViT')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--confidence-threshold', type=float, default=0.95,
                       help='Confidence threshold for filtering examples')
    parser.add_argument('--top-n', type=int, default=1000,
                       help='Top N examples per race to extract')
    parser.add_argument('--output-dir', type=str, default='vit_race_results_small',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("="*80)
    print("ViT RACE CLASSIFIER TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in config.items():
        if key != 'timestamp':
            print(f"  {key}: {value}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load datasets
    print(f"\n{'='*80}")
    print("Loading Datasets")
    print(f"{'='*80}")

    data_dir = Path(args.data_dir)
    train_csv = data_dir / f'vit_train_{args.dataset_type}.csv'
    val_csv = data_dir / f'vit_val_{args.dataset_type}.csv'

    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation data not found: {val_csv}")

    # Load metadata for race names
    metadata_path = data_dir / f'metadata_{args.dataset_type}.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    race_names = metadata['races']
    num_classes = len(race_names)
    print(f"\nClasses: {race_names}")
    print(f"Number of classes: {num_classes}")

    # Create datasets
    train_transform = get_transforms(augment=True, image_size=args.image_size)
    val_transform = get_transforms(augment=False, image_size=args.image_size)

    train_dataset = ChestXRayRaceDataset(train_csv, transform=train_transform)
    val_dataset = ChestXRayRaceDataset(val_csv, transform=val_transform)

    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    print(f"\n{'='*80}")
    print(f"Creating Model: {args.model}")
    print(f"{'='*80}")

    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}\n")

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_acc, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update scheduler
        scheduler.step()

        # Log results
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'race_names': race_names,
            }, output_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get final predictions
    _, _, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device, 'Final')

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=race_names, digits=4))

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(f"{'':>10}", end='')
    for race in race_names:
        print(f"{race:>10}", end='')
    print()
    for i, race in enumerate(race_names):
        print(f"{race:>10}", end='')
        for j in range(len(race_names)):
            print(f"{cm[i][j]:>10}", end='')
        print()

    # Extract predictions with confidence scores
    print(f"\n{'='*80}")
    print("Extracting Confidence Scores")
    print(f"{'='*80}")

    # Run on both train and val sets
    print("\nProcessing validation set...")
    val_predictions = extract_predictions_with_confidence(model, val_loader, device, race_names)
    val_predictions.to_csv(output_dir / 'val_predictions_with_confidence.csv', index=False)
    print(f"Saved: {output_dir / 'val_predictions_with_confidence.csv'}")

    print("\nProcessing training set...")
    train_loader_no_shuffle = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    train_predictions = extract_predictions_with_confidence(model, train_loader_no_shuffle, device, race_names)
    train_predictions.to_csv(output_dir / 'train_predictions_with_confidence.csv', index=False)
    print(f"Saved: {output_dir / 'train_predictions_with_confidence.csv'}")

    # Filter high-confidence examples
    print("\n" + "="*80)
    print("Filtering High-Confidence Examples")
    print("="*80)

    # Combine train and val for filtering
    all_predictions = pd.concat([train_predictions, val_predictions], ignore_index=True)

    filtered_examples = filter_high_confidence_examples(
        all_predictions,
        confidence_threshold=args.confidence_threshold,
        top_n_per_race=args.top_n
    )

    # Save filtered examples
    for race, examples in filtered_examples.items():
        safe_race = race.replace("/", "_").replace(" ", "_")
        output_file = output_dir / f'high_confidence_{safe_race}.csv'
        examples.to_csv(output_file, index=False)
        print(f"\nSaved {len(examples):,} high-confidence {race} examples to: {output_file}")

    # Save summary statistics
    summary = {
        'best_val_acc': best_val_acc,
        'total_images': len(all_predictions),
        'high_confidence_threshold': args.confidence_threshold,
        'high_confidence_counts': {
            race: len(examples) for race, examples in filtered_examples.items()
        },
        'mean_confidence_per_race': {
            race: examples['confidence'].mean() for race, examples in filtered_examples.items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
