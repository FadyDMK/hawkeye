#!/usr/bin/env python3
"""
Script to integrate hard-frame labels into existing YOLO dataset
"""
import os
import shutil
from pathlib import Path

def integrate_hard_frames():
    """Add labeled hard frames to existing training dataset"""
    
    # Paths
    hard_frames_dir = Path("output/hard_frames/project-1-at-2025-09-15-12-29-ffa23dbb")
    dataset_dir = Path("dataset")
    
    hard_images = hard_frames_dir / "images"
    hard_labels = hard_frames_dir / "labels" 
    
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    
    # Create directories if they don't exist
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy images and labels
    copied_count = 0
    
    if hard_images.exists() and hard_labels.exists():
        # Copy all images
        for img_file in hard_images.glob("*.jpg"):
            dest_img = train_images / img_file.name
            if not dest_img.exists():  # Avoid duplicates
                shutil.copy2(img_file, dest_img)
                print(f"Copied image: {img_file.name}")
                copied_count += 1
            
            # Copy corresponding label if it exists
            label_file = hard_labels / (img_file.stem + ".txt")
            if label_file.exists():
                dest_label = train_labels / label_file.name
                if not dest_label.exists():  # Avoid duplicates
                    shutil.copy2(label_file, dest_label)
                    print(f"Copied label: {label_file.name}")
    
    print(f"\n✅ Integration complete!")
    print(f"📊 Added {copied_count} new hard-frame samples to training set")
    
    # Count total training samples
    total_images = len(list(train_images.glob("*.jpg")))
    total_labels = len(list(train_labels.glob("*.txt")))
    
    print(f"📈 Total training samples: {total_images} images, {total_labels} labels")
    
    return copied_count

def verify_dataset_integrity():
    """Verify that all images have corresponding labels"""
    dataset_dir = Path("dataset")
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"
    
    images = list(train_images.glob("*.jpg"))
    labels = list(train_labels.glob("*.txt"))
    
    # Check for missing labels
    missing_labels = []
    for img_file in images:
        label_file = train_labels / (img_file.stem + ".txt")
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} images without labels:")
        for missing in missing_labels[:5]:  # Show first 5
            print(f"   - {missing}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels) - 5} more")
    else:
        print("✅ All images have corresponding labels!")
    
    return len(missing_labels) == 0

if __name__ == "__main__":
    print("🔄 Integrating hard frames into existing dataset...")
    
    # Check if hard frames exist
    hard_frames_path = Path("output/hard_frames/project-1-at-2025-09-15-12-29-ffa23dbb")
    if not hard_frames_path.exists():
        print("❌ Hard frames directory not found!")
        print(f"Expected: {hard_frames_path}")
        exit(1)
    
    # Integrate the frames
    added_count = integrate_hard_frames()
    
    # Verify integrity
    print("\n🔍 Verifying dataset integrity...")
    is_valid = verify_dataset_integrity()
    
    if is_valid and added_count > 0:
        print("\n🚀 Ready for retraining!")
        print("Next steps:")
        print("1. Run: python src/train_yolo.py")
        print("2. Monitor training metrics")
        print("3. Test on validation set")
    elif added_count == 0:
        print("\n💡 No new frames to add (maybe already integrated?)")
    else:
        print("\n⚠️  Dataset has integrity issues - please review before training")
