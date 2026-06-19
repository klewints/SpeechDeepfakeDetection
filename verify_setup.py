#!/usr/bin/env python3
"""
Verification script to ensure all files are in place and imports work.
Run this before starting training.
"""

import os
import sys

def check_file(path, description):
    """Check if file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description:<50} {path}")
    return exists

def main():
    print("\n" + "="*80)
    print(" "*20 + "SPEECH DEEPFAKE DETECTION - SETUP VERIFICATION")
    print("="*80)
    
    # Check directories
    print("\n[1] Checking project structure...")
    dirs = [
        "src",
        "dataset",
        "dataset/audio",
        "dataset/audio/real",
        "dataset/audio/fake",
        "outputs"
    ]
    
    all_dirs_exist = True
    for d in dirs:
        exists = os.path.isdir(d)
        status = "✓" if exists else "✗"
        print(f"  {status} {d:<50}")
        all_dirs_exist = all_dirs_exist and exists
    
    if not all_dirs_exist:
        print("\n⚠ Some directories are missing. Creating them...")
        os.makedirs("dataset/audio/real", exist_ok=True)
        os.makedirs("dataset/audio/fake", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        print("✓ Directories created")
    
    # Check source files
    print("\n[2] Checking source files...")
    src_files = [
        ("src/dataset_loader.py", "CNN dataset loader (original)"),
        ("src/model.py", "CNN model (original)"),
        ("src/wav2vec_dataset.py", "Wav2Vec2 dataset loader (NEW)"),
        ("src/wav2vec_model.py", "Wav2Vec2 model (NEW)"),
    ]
    
    all_src_exist = True
    for path, desc in src_files:
        exists = check_file(path, desc)
        all_src_exist = all_src_exist and exists
    
    # Check training scripts
    print("\n[3] Checking training scripts...")
    train_files = [
        ("train.py", "CNN training script (original)"),
        ("train_wav2vec.py", "Wav2Vec2 training script (NEW)"),
    ]
    
    all_train_exist = True
    for path, desc in train_files:
        exists = check_file(path, desc)
        all_train_exist = all_train_exist and exists
    
    # Check evaluation script
    print("\n[4] Checking evaluation script...")
    eval_files = [
        ("evaluate_models.py", "Model evaluation & comparison (NEW)"),
    ]
    
    all_eval_exist = True
    for path, desc in eval_files:
        exists = check_file(path, desc)
        all_eval_exist = all_eval_exist and exists
    
    # Check dataset
    print("\n[5] Checking dataset...")
    csv_exists = check_file("dataset/metadata.csv", "Dataset metadata CSV")
    
    if csv_exists:
        import pandas as pd
        try:
            df = pd.read_csv("dataset/metadata.csv")
            print(f"      Total samples: {len(df)}")
            print(f"      Columns: {', '.join(df.columns.tolist())}")
            if "label" in df.columns:
                counts = df["label"].value_counts()
                print(f"      Label distribution: {dict(counts)}")
        except Exception as e:
            print(f"      Error reading CSV: {e}")
    
    # Check dependencies
    print("\n[6] Checking Python dependencies...")
    dependencies = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "librosa": "Librosa (audio processing)",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "scikit-learn (metrics)",
        "matplotlib": "Matplotlib (visualization)",
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name:<40} ({module})")
        except ImportError:
            print(f"  ✗ {name:<40} ({module})")
            missing.append(module)
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "VERIFICATION SUMMARY")
    print("="*80)
    
    all_ready = all_src_exist and all_train_exist and all_eval_exist and csv_exists and not missing
    
    if all_ready:
        print("\n✓ All checks passed! Project is ready to run.")
        print("\nNext steps:")
        print("  1. Train CNN model:      python train.py")
        print("  2. Train Wav2Vec2:       python train_wav2vec.py")
        print("  3. Evaluate both models: python evaluate_models.py")
    else:
        print("\n⚠ Some checks failed. Please fix the following:")
        
        if not all_src_exist:
            print("  • Missing source files in src/")
        if not all_train_exist:
            print("  • Missing training scripts")
        if not all_eval_exist:
            print("  • Missing evaluation script")
        if not csv_exists:
            print("  • Missing dataset/metadata.csv")
            print("    → Run: python src/extract_audio.py && python src/create_csv.py")
        if missing:
            print(f"  • Missing dependencies: {', '.join(missing)}")
            print("    → Run: pip install -r requirements_updated.txt")
    
    print("\n" + "="*80 + "\n")
    
    return 0 if all_ready else 1

if __name__ == "__main__":
    sys.exit(main())
