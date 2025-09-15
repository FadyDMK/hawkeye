#!/usr/bin/env python3
"""
Enhanced YOLO training script for retraining with hard frames
"""
from ultralytics import YOLO
import os
from pathlib import Path

def train_enhanced_model():
    """Train YOLO model with enhanced dataset including hard frames"""
    
    # Configuration
    config = {
        'data': 'dataset/data.yaml',  # Path to dataset config
        'epochs': 50,                 # Reduced epochs for fine-tuning
        'imgsz': 640,                # Training image size
        'batch': 16,                 # Batch size (adjust based on GPU memory)
        'lr0': 0.001,               # Initial learning rate (lower for fine-tuning)
        'patience': 10,              # Early stopping patience
        'save_period': 5,            # Save checkpoint every N epochs
        'workers': 4,                # Number of worker threads
        'device': 'cpu',             # Use 'cuda' if you have GPU
        'project': 'runs/detect',    # Project name
        'name': 'retrain_with_hard_frames',  # Run name
    }
    
    print("🚀 Starting enhanced model training...")
    print(f"📊 Configuration: {config}")
    
    # Load pretrained model (use your best existing weights)
    pretrained_weights = "runs/detect/train_finetune_fast416/weights/best.pt"
    if not os.path.exists(pretrained_weights):
        print(f"⚠️  Pretrained weights not found: {pretrained_weights}")
        print("🔄 Using YOLOv8n base model instead")
        pretrained_weights = "yolov8n.pt"
    
    # Initialize model
    model = YOLO(pretrained_weights)
    
    # Train the model
    try:
        results = model.train(**config)
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Validate the model
        print("🔍 Running validation...")
        val_results = model.val()
        
        print(f"📊 Validation mAP50: {val_results.box.map50:.4f}")
        print(f"📊 Validation mAP50-95: {val_results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def compare_models():
    """Compare old vs new model performance"""
    
    old_model_path = "runs/detect/train_finetune_fast416/weights/best.pt"
    new_model_path = "runs/detect/retrain_with_hard_frames/weights/best.pt"
    
    if os.path.exists(old_model_path) and os.path.exists(new_model_path):
        print("\n📊 Model Comparison:")
        
        # Load both models
        old_model = YOLO(old_model_path)
        new_model = YOLO(new_model_path)
        
        # Validate both on test set
        print("🔍 Validating old model...")
        old_results = old_model.val()
        
        print("🔍 Validating new model...")
        new_results = new_model.val()
        
        # Compare metrics
        old_map50 = old_results.box.map50
        new_map50 = new_results.box.map50
        improvement = new_map50 - old_map50
        
        print(f"\n📈 Results Comparison:")
        print(f"Old model mAP50: {old_map50:.4f}")
        print(f"New model mAP50: {new_map50:.4f}")
        print(f"Improvement: {improvement:+.4f} ({improvement/old_map50*100:+.2f}%)")
        
        if improvement > 0:
            print("🎉 New model is better! Consider updating your config.")
        else:
            print("🤔 Old model still better. May need more training or different hyperparams.")
    
    else:
        print("⚠️  Cannot compare - missing model files")

if __name__ == "__main__":
    print("🔄 Enhanced YOLO Training with Hard Frames")
    print("=" * 50)
    
    # Check dataset
    if not os.path.exists("dataset/data.yaml"):
        print("❌ Dataset config not found: dataset/data.yaml")
        exit(1)
    
    # Start training
    results = train_enhanced_model()
    
    if results:
        # Compare with old model
        compare_models()
        
        print("\n🎯 Training Summary:")
        print("1. ✅ Added 88 hard frames to dataset")
        print("2. ✅ Retrained model with enhanced data")
        print("3. ✅ Validated performance")
        print("4. 🔄 Ready to update detection config if improved")
    else:
        print("\n❌ Training failed - please check logs and try again")
