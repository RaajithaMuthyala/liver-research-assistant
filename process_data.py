import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import json

class LiverSDOHProcessor:
    def __init__(self):
        # Terms from your PubMed search - EXACTLY what you used
        self.sdoh_terms = [
            'social determinants', 'safe housing', 'house', 'housing', 'transportation', 
            'neighborhood', 'racism', 'discrimination', 'violence', 'education', 
            'job opportunities', 'job', 'employment', 'income', 'nutritious food', 'food', 
            'physical activity', 'polluted air', 'pollution', 'polluted water', 'water', 
            'language', 'literacy', 'socioeconomic', 'poverty', 'access', 'insurance'
        ]
        
        self.liver_terms = [
            'liver', 'hepatic', 'liver diseases', 'liver disease', 'hepatic conditions',
            'hcc', 'hepatocellular carcinoma', 'cirrhosis', 'fibrosis', 'steatosis',
            'nash', 'nafld', 'masld', 'viral hepatitis', 'hbv', 'hcv', 'hepatitis',
            'liver cancer', 'liver failure', 'portal hypertension', 'ascites'
        ]
        
        self.risk_factors = [
            'diabetes', 'type 2 diabetes', 'obesity', 'metabolic syndrome', 'alcohol', 
            'tobacco', 'smoking', 'aflatoxin', 'air pollution', 'chemical exposure',
            'diet', 'lifestyle', 'metabolic dysfunction', 'insulin resistance'
        ]

    def create_training_data(self, df):
        """Creates question-answer pairs for training"""
        training_examples = []
        
        print(f"Processing {len(df)} abstracts...")
        
        # Question templates based on research patterns
        templates = [
            "How do social determinants affect liver disease?",
            "What factors influence liver disease outcomes?",
            "How does socioeconomic status impact liver health?",
            "What are the environmental risks for liver disease?",
            "How do lifestyle factors affect liver conditions?",
            "What social factors contribute to liver disease disparities?",
            "How does access to healthcare affect liver disease treatment?"
        ]
        
        for idx, row in df.iterrows():
            abstract = str(row['Abstract']) if pd.notna(row['Abstract']) else ""
            doi = str(row['DOI']) if pd.notna(row['DOI']) else ""
            
            if len(abstract) < 50:  # Skip very short abstracts
                continue
                
            # Find domain terms in this abstract
            abstract_lower = abstract.lower()
            found_sdoh = [term for term in self.sdoh_terms if term in abstract_lower]
            found_liver = [term for term in self.liver_terms if term in abstract_lower]
            found_risk = [term for term in self.risk_factors if term in abstract_lower]
            
            # Only use abstracts that have both SDOH and liver terms
            if len(found_sdoh) >= 1 and len(found_liver) >= 1:
                
                # Create training examples
                for template in templates[:3]:  # Use first 3 templates
                    
                    # Create a short summary instead of full abstract
                    summary = self.create_summary(abstract, found_sdoh, found_liver, found_risk)
                    
                    training_example = {
                        "instruction": "Answer the research question about liver disease and social determinants of health based on scientific evidence.",
                        "input": template,
                        "output": f"{summary} [DOI: {doi}]"
                    }
                    
                    training_examples.append(training_example)
        
        print(f"Created {len(training_examples)} training examples")
        return training_examples
    
    def create_summary(self, abstract, sdoh_terms, liver_terms, risk_terms):
        """Creates a 1-2 sentence summary focusing on key relationships"""
        
        # Extract key sentences that mention both SDOH and liver terms
        sentences = abstract.split('.')
        key_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_sdoh = any(term in sentence_lower for term in sdoh_terms)
            has_liver = any(term in sentence_lower for term in liver_terms)
            
            if has_sdoh and has_liver:
                key_sentences.append(sentence.strip())
        
        # If we found connecting sentences, use them
        if key_sentences:
            summary = '. '.join(key_sentences[:2])  # Take first 2 connecting sentences
        else:
            # Create a summary highlighting the main findings
            summary = f"Research shows connections between {', '.join(sdoh_terms[:2])} and {', '.join(liver_terms[:2])}"
            if risk_terms:
                summary += f" with {', '.join(risk_terms[:2])} as key factors"
        
        return summary

# Main processing
def main():
    print("Starting data processing...")
    
    # Load your data - EXACT filename
    try:
        df = pd.read_csv('abstractanddoi.csv')
        print(f"Successfully loaded {len(df)} abstracts")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("ERROR: abstractanddoi.csv not found!")
        print("Make sure the file is in the same folder as this script")
        return
    
    # Initialize processor
    processor = LiverSDOHProcessor()
    
    # Create training data
    training_data = processor.create_training_data(df)
    
    # Save training data
    with open('training_data.jsonl', 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved {len(training_data)} training examples to training_data.jsonl")
    print("Data processing complete!")

if __name__ == "__main__":
    main()
