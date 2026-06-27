"""
Large-scale dataset generation script.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.data.dataset_generator import DatasetGenerator
from src.data.dataset import split_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate large dataset")
    
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of samples to generate (default: 2000)")
    parser.add_argument("--output_path", type=str, default="data/raw/large_dataset.jsonl",
                        help="Output file path")
    parser.add_argument("--high_ratio", type=float, default=0.3,
                        help="Ratio of high confidence samples")
    parser.add_argument("--medium_ratio", type=float, default=0.4,
                        help="Ratio of medium confidence samples")
    parser.add_argument("--low_ratio", type=float, default=0.3,
                        help="Ratio of low confidence samples")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Ratio of training samples")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Ratio of validation samples")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Ratio of test samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print("=" * 60)
    print("Large Dataset Generation")
    print("=" * 60)
    print(f"\nNumber of samples: {args.num_samples}")
    print(f"High confidence ratio: {args.high_ratio}")
    print(f"Medium confidence ratio: {args.medium_ratio}")
    print(f"Low confidence ratio: {args.low_ratio}")
    print(f"Output path: {args.output_path}")

    # Create dataset generator
    generator = DatasetGenerator(seed=args.seed)

    # Generate dataset
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        high_ratio=args.high_ratio,
        medium_ratio=args.medium_ratio,
        low_ratio=args.low_ratio
    )
    
    # Display statistics
    categories = {}
    difficulties = {}
    confidence_ranges = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0
    }
    
    for sample in samples:
        cat = sample['category']
        diff = sample['difficulty']
        conf = sample['confidence_label']
        
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        if conf < 0.2:
            confidence_ranges["0.0-0.2"] += 1
        elif conf < 0.4:
            confidence_ranges["0.2-0.4"] += 1
        elif conf < 0.6:
            confidence_ranges["0.4-0.6"] += 1
        elif conf < 0.8:
            confidence_ranges["0.6-0.8"] += 1
        else:
            confidence_ranges["0.8-1.0"] += 1
    
    print(f"\nGenerated {len(samples)} samples")
    
    print("\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} ({count/len(samples)*100:.1f}%)")
    
    print("\nDifficulties:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count} ({count/len(samples)*100:.1f}%)")
    
    print("\nConfidence ranges:")
    for range_name, count in confidence_ranges.items():
        print(f"  {range_name}: {count} ({count/len(samples)*100:.1f}%)")
    
    # Display sample examples
    print("\nSample examples:")
    for i in range(min(5, len(samples))):
        sample = samples[i]
        print(f"\n{i+1}. {sample['question']}")
        print(f"   Answer: {sample['answer']}")
        print(f"   Confidence: {sample['confidence_label']:.3f}")
        print(f"   Category: {sample['category']}, Difficulty: {sample['difficulty']}")
    
    # Save
    output_path = Path(args.output_path)
    generator.save_dataset(samples, output_path)
    print(f"\nDataset saved to {output_path}")
    
    # Split dataset
    print("\nSplitting dataset...")
    output_dir = output_path.parent.parent / "processed"
    splits = split_dataset(
        output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=output_dir,
        seed=args.seed
    )
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/len(samples)*100:.1f}%)")
    print(f"  Val: {len(splits['val'])} samples ({len(splits['val'])/len(samples)*100:.1f}%)")
    print(f"  Test: {len(splits['test'])} samples ({len(splits['test'])/len(samples)*100:.1f}%)")
    
    print(f"\n{'=' * 60}")
    print("Dataset generation completed!")
    print(f"Files saved to:")
    print(f"  Raw: {output_path}")
    print(f"  Processed: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
