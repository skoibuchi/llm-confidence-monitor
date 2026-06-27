"""
Dataset - dataset management for knowledge probing.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeProbeDataset(Dataset):
    """
    Dataset for knowledge probing.

    Loads questions and confidence labels from a JSONL file.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        return_text: bool = False
    ):
        """
        Args:
            data_path: Path to the data file (.jsonl format)
            tokenizer: Tokenizer
            max_length: Maximum token length
            return_text: Whether to also return text
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_text = return_text
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """
        Load data from a JSONL file.

        Returns:
            List[Dict]: List of data items
        """
        data = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return data
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        
        return data
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        """
        Retrieve a data item.

        Args:
            idx: Index

        Returns:
            Dict: Data item containing:
                - input_ids: Token IDs (max_length,)
                - attention_mask: Attention mask (max_length,)
                - label: Confidence label (scalar)
                - question: Question text (if return_text=True)
                - answer: Answer (if return_text=True)
        """
        item = self.data[idx]
        
        # Tokenize the question
        question = item['question']
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label (confidence)
        label = torch.tensor(item['confidence_label'], dtype=torch.float32)
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
        
        # Optionally return raw text
        if self.return_text:
            result['question'] = question
            result['answer'] = item.get('answer', '')
            result['category'] = item.get('category', '')
        
        return result
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Compute dataset statistics.

        Returns:
            Dict: Statistics dictionary
        """
        if len(self.data) == 0:
            return {}
        
        # Confidence statistics
        labels = [item['confidence_label'] for item in self.data]

        # Category distribution
        categories = {}
        for item in self.data:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'num_samples': len(self.data),
            'label_mean': sum(labels) / len(labels),
            'label_min': min(labels),
            'label_max': max(labels),
            'categories': categories
        }


class KnowledgeDataset:
    """
    Simple dataset class (for visualization).
    Holds a list of questions and corresponding labels.
    """

    def __init__(self, questions: List[str], labels: List[float]):
        """
        Args:
            questions: List of questions
            labels: List of confidence labels
        """
        self.questions = questions
        self.labels = labels
        
        if len(questions) != len(labels):
            raise ValueError("questions and labels must have the same length")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]
    
    @classmethod
    def load(cls, data_path: Union[str, Path]) -> 'KnowledgeDataset':
        """
        Load a dataset from a JSONL file.

        Args:
            data_path: Path to the data file

        Returns:
            KnowledgeDataset: Loaded dataset
        """
        data_path = Path(data_path)
        
        questions = []
        labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                questions.append(item['question'])
                labels.append(item['confidence_label'])
        
        return cls(questions, labels)
    
    def save(self, output_path: Union[str, Path]):
        """
        Save the dataset to a JSONL file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for question, label in zip(self.questions, self.labels):
                item = {
                    'question': question,
                    'confidence_label': label
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_sample_dataset(
    output_path: Union[str, Path],
    num_samples: int = 100
):
    """
    Create a sample dataset.

    Args:
        output_path: Output file path
        num_samples: Number of samples
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sample data
    samples = []

    # High-confidence (30%)
    high_confidence_questions = [
        ("What is the capital of Japan?", "Tokyo", 0.98, "geography", "easy"),
        ("What is the chemical formula of water?", "H2O", 0.99, "science", "easy"),
        ("Does the Earth orbit the Sun?", "Yes", 0.99, "science", "easy"),
        ("What is 1 + 1?", "2", 1.0, "math", "easy"),
        ("Is Mount Fuji located in Japan?", "Yes", 0.98, "geography", "easy"),
    ]

    # Medium-confidence (40%)
    medium_confidence_questions = [
        ("How do you reverse a list in Python?", "reverse() or slice [::-1]", 0.65, "technology", "medium"),
        ("In what year was Napoleon born?", "1769", 0.55, "history", "medium"),
        ("What is the speed of light in km/s?", "approximately 300,000 km/s", 0.70, "science", "medium"),
        ("What is the approximate population of Japan?", "approximately 120 million", 0.60, "geography", "medium"),
        ("What are Shakespeare's major works?", "Hamlet, Romeo and Juliet, etc.", 0.68, "literature", "medium"),
    ]

    # Low-confidence (30%)
    low_confidence_questions = [
        ("What is the gravitational acceleration on planet X in m/s²?", "non-existent planet", 0.05, "fictional", "hard"),
        ("Who won the 2025 Nobel Prize in Physics?", "future information", 0.10, "current_events", "hard"),
        ("What is the atomic number of fictional element Z?", "non-existent element", 0.02, "fictional", "hard"),
        ("What is the name of Mars's 5th moon?", "Mars has only two moons", 0.08, "fictional", "hard"),
        ("What is the latest AI technology XYZ?", "fictional technology", 0.15, "fictional", "hard"),
    ]
    
    # Generate samples
    import random
    random.seed(42)
    
    all_questions = (
        high_confidence_questions * 6 +  # 30 samples
        medium_confidence_questions * 8 +  # 40 samples
        low_confidence_questions * 6  # 30 samples
    )
    
    random.shuffle(all_questions)
    all_questions = all_questions[:num_samples]
    
    for i, (question, answer, confidence, category, difficulty) in enumerate(all_questions):
        sample = {
            "id": f"sample_{i:03d}",
            "question": question,
            "answer": answer,
            "confidence_label": confidence,
            "category": category,
            "difficulty": difficulty,
            "metadata": {
                "source": "manual",
                "created_at": "2024-01-01"
            }
        }
        samples.append(sample)
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Created sample dataset with {len(samples)} samples at {output_path}")
    
    return samples


def split_dataset(
    data_path: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42
):
    """
    Split a dataset into train / validation / test sets.

    Args:
        data_path: Path to the source data file
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        test_ratio: Proportion of test data
        output_dir: Output directory (defaults to the source file's directory)
        seed: Random seed
    """
    import random
    
    data_path = Path(data_path)
    output_dir = Path(output_dir) if output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Shuffle
    random.seed(seed)
    random.shuffle(data)
    
    # Split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(split_data)} samples to {output_path}")
    
    return splits


def main():
    """Main function for testing."""
    from transformers import AutoTokenizer
    
    print("=== Test 1: Create sample dataset ===")
    
    # Create a sample dataset
    output_path = "data/raw/sample_dataset.jsonl"
    samples = create_sample_dataset(output_path, num_samples=100)
    print(f"Created {len(samples)} samples")
    
    print("\n=== Test 2: Load dataset ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = KnowledgeProbeDataset(
        output_path,
        tokenizer,
        return_text=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Statistics
    stats = dataset.get_statistics()
    print("\nDataset statistics:")
    print(f"  Number of samples: {stats['num_samples']}")
    print(f"  Label mean: {stats['label_mean']:.3f}")
    print(f"  Label range: [{stats['label_min']:.3f}, {stats['label_max']:.3f}]")
    print(f"  Categories: {stats['categories']}")
    
    # Display sample items
    print("\n=== Test 3: Sample items ===")
    for i in range(3):
        item = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Question: {item['question']}")
        print(f"  Answer: {item['answer']}")
        print(f"  Label: {item['label'].item():.3f}")
        print(f"  Category: {item['category']}")
        print(f"  Input IDs shape: {item['input_ids'].shape}")
    
    print("\n=== Test 4: Split dataset ===")
    
    # Split dataset
    splits = split_dataset(
        output_path,
        output_dir="data/processed"
    )
    
    print(f"Train: {len(splits['train'])} samples")
    print(f"Val: {len(splits['val'])} samples")
    print(f"Test: {len(splits['test'])} samples")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
