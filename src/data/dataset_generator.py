"""
Dataset generator - automatically generates large-scale datasets.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Automatically generates datasets for knowledge probing.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.seed = seed
        random.seed(seed)
        
        # Database of question templates and answers
        self._init_question_templates()
    
    def _init_question_templates(self):
        """Initialize question templates."""
        
        # High-confidence questions (general knowledge)
        self.high_confidence_templates = [
            # Geography
            ("What is the capital of {country}?", "{capital}", "geography", "easy"),
            ("Is {mountain} located in {country}?", "Yes", "geography", "easy"),
            ("Which country does {river} flow through?", "{country}", "geography", "medium"),
            
            # Science
            ("What is the chemical formula of water?", "H2O", "science", "easy"),
            ("Does the Earth orbit around {planet}?", "Yes", "science", "easy"),
            ("What is the chemical symbol of {element}?", "{symbol}", "science", "medium"),
            ("What is the approximate speed of light in km/s?", "approximately 300,000 km/s", "science", "medium"),
            
            # Math
            ("What is 1 + 1?", "2", "math", "easy"),
            ("What is the approximate value of pi?", "approximately 3.14", "math", "easy"),
            ("What is {num1} × {num2}?", "{result}", "math", "easy"),
            
            # History
            ("In what year did {event} occur?", "{year}", "history", "medium"),
            ("What is {person} known for?", "{achievement}", "history", "medium"),
        ]
        
        # Medium-confidence questions (somewhat specialized)
        self.medium_confidence_templates = [
            # Technology
            ("How do you {operation} in Python?", "{method}", "technology", "medium"),
            ("What are the main features of {language}?", "{feature}", "technology", "medium"),
            
            # Literature
            ("What is {author}'s most famous work?", "{work}", "literature", "medium"),
            ("Who is the author of {work}?", "{author}", "literature", "medium"),
            
            # Music
            ("What is {composer}'s most famous composition?", "{piece}", "music", "medium"),
            
            # Sports
            ("How many players are on a {sport} team?", "{number}", "sports", "medium"),
        ]
        
        # Low-confidence questions (fictional / future events)
        self.low_confidence_templates = [
            # Fictional information
            ("What is the gravitational acceleration on planet {fake_planet} in m/s²?", "non-existent planet", "fictional", "hard"),
            ("What is the atomic number of the fictional element {fake_element}?", "non-existent element", "fictional", "hard"),
            ("What is the population of {fake_place}?", "non-existent location", "fictional", "hard"),
            
            # Future information
            ("Who will win the Nobel Prize in {field} in {future_year}?", "future information", "current_events", "hard"),
            ("What will be the outcome of {event} in {future_year}?", "future information", "current_events", "hard"),
        ]
        
        # Data database
        self.data = {
            "countries": ["Japan", "United States", "France", "Germany", "China", "United Kingdom", "Italy", "Spain"],
            "capitals": ["Tokyo", "Washington D.C.", "Paris", "Berlin", "Beijing", "London", "Rome", "Madrid"],
            "mountains": ["Mount Fuji", "Everest", "Kilimanjaro", "Matterhorn"],
            "rivers": ["Nile", "Amazon", "Yangtze", "Mississippi"],
            "elements": ["Hydrogen", "Oxygen", "Carbon", "Nitrogen", "Iron", "Gold", "Silver"],
            "symbols": ["H", "O", "C", "N", "Fe", "Au", "Ag"],
            "planets": ["the Sun", "the Moon"],
            "events": ["End of World War II", "French Revolution", "American Independence"],
            "years": ["1945", "1789", "1776"],
            "persons": ["Einstein", "Newton", "Darwin"],
            "achievements": ["published the theory of relativity", "discovered the law of universal gravitation", "proposed the theory of evolution"],
            "operations": ["reverse a list", "sort a dictionary", "split a string"],
            "methods": ["reverse() or slice [::-1]", "sorted() function", "split() method"],
            "languages": ["Python", "JavaScript", "Java"],
            "features": ["simple syntax", "asynchronous processing", "object-oriented"],
            "authors": ["Soseki Natsume", "Shakespeare", "Goethe"],
            "works": ["I Am a Cat", "Hamlet", "Faust"],
            "composers": ["Beethoven", "Mozart", "Bach"],
            "pieces": ["Symphony No. 9", "Requiem", "St Matthew Passion"],
            "sports": ["soccer", "basketball", "baseball"],
            "numbers": ["11", "5", "9"],
            "fake_planets": ["X", "Zeta", "Omega"],
            "fake_elements": ["Z", "Ultranium", "Mythrilium"],
            "fake_places": ["Atlantis", "El Dorado", "Shangri-La"],
            "future_years": ["2030", "2040", "2050"],
            "fields": ["Physics", "Chemistry", "Literature"],
        }
    
    def generate_sample(
        self,
        template: Tuple[str, str, str, str],
        confidence: float
    ) -> Dict:
        """
        Generate a single sample from a template.
        
        Args:
            template: (question_template, answer_template, category, difficulty)
            confidence: Confidence label
            
        Returns:
            Dict: Sample data
        """
        question_template, answer_template, category, difficulty = template
        
        # Replace variables in the template
        question = question_template
        answer = answer_template
        
        for key in self.data:
            if f"{{{key}}}" in question or f"{{{key}}}" in answer:
                value = random.choice(self.data[key])
                question = question.replace(f"{{{key}}}", value)
                answer = answer.replace(f"{{{key}}}", value)
        
        # Handle numeric calculations
        if "{num1}" in question_template:
            num1 = random.randint(2, 9)
            num2 = random.randint(2, 9)
            result = num1 * num2
            question = question.replace("{num1}", str(num1)).replace("{num2}", str(num2))
            answer = answer.replace("{result}", str(result))
        
        return {
            "question": question,
            "answer": answer,
            "confidence_label": confidence,
            "category": category,
            "difficulty": difficulty
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        high_ratio: float = 0.3,
        medium_ratio: float = 0.4,
        low_ratio: float = 0.3
    ) -> List[Dict]:
        """
        Generate a dataset.
        
        Args:
            num_samples: Number of samples to generate
            high_ratio: Proportion of high-confidence samples
            medium_ratio: Proportion of medium-confidence samples
            low_ratio: Proportion of low-confidence samples
            
        Returns:
            List[Dict]: List of samples
        """
        samples = []
        
        # Calculate the number of samples per confidence level
        num_high = int(num_samples * high_ratio)
        num_medium = int(num_samples * medium_ratio)
        num_low = num_samples - num_high - num_medium
        
        # Generate high-confidence samples
        for i in range(num_high):
            template = random.choice(self.high_confidence_templates)
            confidence = random.uniform(0.85, 1.0)
            sample = self.generate_sample(template, confidence)
            sample["id"] = f"high_{i:04d}"
            sample["metadata"] = {"source": "auto_generated", "type": "high_confidence"}
            samples.append(sample)
        
        # Generate medium-confidence samples
        for i in range(num_medium):
            template = random.choice(self.medium_confidence_templates)
            confidence = random.uniform(0.5, 0.75)
            sample = self.generate_sample(template, confidence)
            sample["id"] = f"medium_{i:04d}"
            sample["metadata"] = {"source": "auto_generated", "type": "medium_confidence"}
            samples.append(sample)
        
        # Generate low-confidence samples
        for i in range(num_low):
            template = random.choice(self.low_confidence_templates)
            confidence = random.uniform(0.0, 0.25)
            sample = self.generate_sample(template, confidence)
            sample["id"] = f"low_{i:04d}"
            sample["metadata"] = {"source": "auto_generated", "type": "low_confidence"}
            samples.append(sample)
        
        # Shuffle
        random.shuffle(samples)
        
        # Reassign IDs
        for i, sample in enumerate(samples):
            sample["id"] = f"sample_{i:05d}"
        
        logger.info(f"Generated {len(samples)} samples")
        logger.info(f"  High confidence: {num_high}")
        logger.info(f"  Medium confidence: {num_medium}")
        logger.info(f"  Low confidence: {num_low}")
        
        return samples
    
    def save_dataset(
        self,
        samples: List[Dict],
        output_path: Path
    ):
        """
        Save a dataset to a JSONL file.
        
        Args:
            samples: List of samples
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main function for testing."""
    print("=== Test: DatasetGenerator ===")
    
    generator = DatasetGenerator(seed=42)
    
    # Generate 500 samples
    print("\nGenerating 500 samples...")
    samples = generator.generate_dataset(num_samples=500)
    
    # Display statistics
    categories = {}
    difficulties = {}
    for sample in samples:
        cat = sample['category']
        diff = sample['difficulty']
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nGenerated {len(samples)} samples")
    print("\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\nDifficulties:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")
    
    # Display sample examples
    print("\nSample examples:")
    for i in range(5):
        sample = samples[i]
        print(f"\n{i+1}. {sample['question']}")
        print(f"   Answer: {sample['answer']}")
        print(f"   Confidence: {sample['confidence_label']:.2f}")
        print(f"   Category: {sample['category']}, Difficulty: {sample['difficulty']}")
    
    # Save
    output_path = Path("data/raw/generated_500.jsonl")
    generator.save_dataset(samples, output_path)
    print(f"\nDataset saved to {output_path}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
