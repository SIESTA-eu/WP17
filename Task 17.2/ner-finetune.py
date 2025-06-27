#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

"""
BERT NER Fine-tuning Pipeline for Cybersecurity Text
Complete pipeline for fine-tuning BERT on custom NER data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast, 
    BertForTokenClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
import json
import re
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import warnings
import random
# warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class NERDataset(Dataset):
    """Custom dataset for NER fine-tuning"""
    
    # Update NERDataset __init__ to include all needed attributes
    def __init__(self, texts, labels, tokenizer, label2id, use_label_mapping=True, cecilia_to_conll=None, max_length=512):
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.use_label_mapping = use_label_mapping
        self.cecilia_to_conll = cecilia_to_conll or {}
    
    # Add the mapping method to NERDataset
    def map_cecilia_to_conll(self, cecilia_label):
        """Map CECILIA label to CoNLL-03 equivalent"""
        return self.cecilia_to_conll.get(cecilia_label, 'MISC')
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize and align labels
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            is_split_into_words=False
        )
        
        # Align labels with tokens
        aligned_labels = self.align_labels_with_tokens(
            text, labels, encoding['offset_mapping'].squeeze().tolist()
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def align_labels_with_tokens(self, text, entities, offset_mapping):
       """Align entity labels with BERT tokens using IOB2 format"""
       # Initialize all tokens as 'O' (Outside) - use label IDs, not strings
       labels = [self.label2id['O']] * len(offset_mapping)
       
       for entity in entities:
           start_char, end_char, entity_text, entity_labels = entity
           entity_type = entity_labels[0] if entity_labels else 'O'
           
           # Skip punctuation-only entities
           if entity_type == '0':
               continue
           
           # Map CECILIA to CoNLL-03 if using label mapping
           if self.use_label_mapping:
               entity_type = self.map_cecilia_to_conll(entity_type)
               
           # Find tokens that overlap with entity
           entity_token_start = None
           entity_token_end = None
           
           for i, (token_start, token_end) in enumerate(offset_mapping):
               # Skip special tokens ([CLS], [SEP], [PAD])
               if token_start == 0 and token_end == 0:
                   continue
                   
               # Check if token overlaps with entity
               if token_start < end_char and token_end > start_char:
                   if entity_token_start is None:
                       entity_token_start = i
                   entity_token_end = i
           
           # Assign IOB2 labels using label IDs
           if entity_token_start is not None and entity_token_end is not None:
               b_label = f'B-{entity_type}'
               i_label = f'I-{entity_type}'
               
               # Check if labels exist in our label set
               if b_label in self.label2id:
                   labels[entity_token_start] = self.label2id[b_label]
                   for i in range(entity_token_start + 1, entity_token_end + 1):
                       if i_label in self.label2id:
                           labels[i] = self.label2id[i_label]
       
       return labels

class NERFineTuner:
    """Main class for NER fine-tuning pipeline"""
    
    def __init__(self, model_name="dslim/bert-base-NER", use_label_mapping=True):
        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = None
        self.use_label_mapping = use_label_mapping
        
        # CECILIA to CoNLL-03 mapping
        self.cecilia_to_conll = {
            'Person': 'PER',
            'Norp': 'MISC', 
            'Facilities': 'LOC',
            'GPE_Location': 'LOC',
            'Location': 'LOC', 
            'Organization': 'ORG',
            'Group': 'ORG',
            'Date': 'MISC',  # CoNLL-03 doesn't have DATE, use MISC
            'Time': 'MISC',
            'Money': 'MISC', 
            'Quantities': 'MISC',
            'Numbers_O': 'MISC',
            'Numbers_C': 'MISC',
            'URL': 'MISC',
            'Emails': 'MISC', 
            'Phone': 'MISC',
            'Address': 'LOC',
            'IP': 'MISC'
        }
        
        if use_label_mapping:
            # Use CoNLL-03 labels (9 labels)
            self.label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        else:
            self.label_list = []
            
        self.label2id = {}
        self.id2label = {}

    def map_cecilia_to_conll(self, cecilia_label):
        """Map CECILIA label to CoNLL-03 equivalent"""
        return self.cecilia_to_conll.get(cecilia_label, 'MISC')
    
    def extract_annotations(self, annotation_data):
        """Extract annotations from JSON string format"""
        
        # Handle pandas Series (duplicate column names)
        if hasattr(annotation_data, 'values'):
            # Look for the JSON string in the Series values
            for value in annotation_data.values:
                value_str = str(value).strip()
                if value_str.startswith('[') and value_str.endswith(']'):
                    annotation_str = value_str
                    break
            else:
                return []
        else:
            annotation_str = str(annotation_data).strip()
        
        # Basic validation
        if not annotation_str or annotation_str in ['nan', 'None', 'x', '']:
            return []
        
        try:
            # Parse JSON directly
            annotations = json.loads(annotation_str)
            if not isinstance(annotations, list):
                return []
                
            processed_annotations = []
            for annotation in annotations:
                if isinstance(annotation, dict) and all(k in annotation for k in ['start', 'end', 'text', 'labels']):
                    start = annotation["start"]
                    end = annotation["end"] 
                    text = annotation["text"]
                    labels = annotation["labels"]
                    processed_annotations.append((start, end, text, labels))
            
            return processed_annotations
        except Exception as e:
            return []
    
    def prepare_data(self, df, text_column='TEXT', annotation_column='INFO'):
        """Prepare data for training"""
        print("Preparing data...")
        print(f"Original dataset size: {len(df)}")
        
        # Debug: Check what's in the INFO column
        print(f"Sample INFO values:")
        for i in range(min(3, len(df))):
            print(f"  Row {i}: {repr(df.iloc[i][annotation_column])}")
            print(f"  Type: {type(df.iloc[i][annotation_column])}")
        
        # Create a copy first
        df = df.copy()
        
        # Extract annotations using a manual loop
        annotations_list = []
        successful_extractions = 0
        for idx, row in df.iterrows():
            annotations = self.extract_annotations(row[annotation_column])
            annotations_list.append(annotations)
            if len(annotations) > 0:
                successful_extractions += 1
        
        print(f"Successfully extracted annotations from {successful_extractions} rows")
        
        # Assign as a simple list
        df['annotations'] = annotations_list
        
        # Filter out rows without annotations
        df_filtered = df[df['annotations'].apply(len) > 0].copy()
        print(f"Filtered dataset: {len(df_filtered)} rows with annotations out of {len(df)} total")
        
        # If we have no data, let's see what went wrong
        if len(df_filtered) == 0:
            print("ERROR: No rows with valid annotations found!")
            print("Sample annotation extraction results:")
            for i in range(min(5, len(df))):
                sample_result = self.extract_annotations(df.iloc[i][annotation_column])
                print(f"  Row {i}: {sample_result}")
            return df_filtered  # Return empty dataframe to see the error
        
        if self.use_label_mapping:
            # Use pre-defined CoNLL-03 labels (no need to extract from data)
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            print(f"Using CoNLL-03 label mapping with {len(self.label_list)} labels: {self.label_list}")
            
            # Show the mapping being used
            print("CECILIA to CoNLL-03 mapping:")
            unique_cecilia_labels = set()
            for annotations in df_filtered['annotations']:
                for _, _, _, labels in annotations:
                    for label in labels:
                        if label != '0':
                            unique_cecilia_labels.add(label)
            
            for cecilia_label in sorted(unique_cecilia_labels):
                conll_label = self.map_cecilia_to_conll(cecilia_label)
                print(f"  {cecilia_label} -> {conll_label}")
                
        else:
            # Extract all unique labels from data (original approach)
            all_labels = set(['O'])  # Start with 'O' for outside
            for annotations in df_filtered['annotations']:
                for _, _, _, labels in annotations:
                    for label in labels:
                        if label != '0':  # Skip punctuation labels
                            all_labels.add(f'B-{label}')
                            all_labels.add(f'I-{label}')
            
            self.label_list = sorted(list(all_labels))
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
            self.id2label = {i: label for label, i in self.label2id.items()}
            print(f"Extracted {len(self.label_list)} unique labels from data: {self.label_list}")
        
        return df_filtered

    # Update create_datasets to pass all needed parameters
    def create_datasets(self, df, text_column='TEXT', test_size=0.2, val_size=0.1):
        """Create train, validation, and test datasets"""
        print("Creating datasets...")
        
        texts = df[text_column].tolist()
        annotations = df['annotations'].tolist()
        
        # First split: separate test set
        train_texts, test_texts, train_annotations, test_annotations = train_test_split(
            texts, annotations, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: separate validation from training
        train_texts, val_texts, train_annotations, val_annotations = train_test_split(
            train_texts, train_annotations, test_size=val_size/(1-test_size), random_state=42
        )
        
        print(f"Dataset splits:")
        print(f"  Training: {len(train_texts)} samples")
        print(f"  Validation: {len(val_texts)} samples") 
        print(f"  Test: {len(test_texts)} samples")
        
        # Create datasets - pass all needed parameters
        train_dataset = NERDataset(train_texts, train_annotations, self.tokenizer, self.label2id, self.use_label_mapping, self.cecilia_to_conll)
        val_dataset = NERDataset(val_texts, val_annotations, self.tokenizer, self.label2id, self.use_label_mapping, self.cecilia_to_conll)
        test_dataset = NERDataset(test_texts, test_annotations, self.tokenizer, self.label2id, self.use_label_mapping, self.cecilia_to_conll)
        
        return train_dataset, val_dataset, test_dataset
        
    
    def initialize_model(self):
        """Initialize the model with correct number of labels"""
        print("Initializing model...")
        self.model = BertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(device)
    
    def train_manual(self, train_dataset, val_dataset, output_dir="./bert-base-ner-finetuned"):
        """Manual training loop without Trainer"""
        print("Starting manual fine-tuning...")
        
        # Import tqdm
        from tqdm import tqdm
        
        # Training parameters
        num_epochs = 8
        batch_size = 16
        learning_rate = 5e-5
        warmup_steps = 500
        
        # Create data loaders
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForTokenClassification
        
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            label_pad_token_id=-100
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=data_collator
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=data_collator
        )
        
        # Setup optimizer and scheduler
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop with progress bars
        self.model.train()
        best_val_loss = float('inf')
        
        # Epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            total_train_loss = 0
            
            # Training step progress bar
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
            
            for step, batch in enumerate(train_pbar):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
                
                # Update progress bar with current loss
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_train_loss/(step+1):.4f}'
                })
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation with progress bar
            self.model.eval()
            total_val_loss = 0
            
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            
            with torch.no_grad():
                for batch in val_pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    total_val_loss += outputs.loss.item()
                    
                    val_pbar.set_postfix({
                        'val_loss': f'{outputs.loss.item():.4f}'
                    })
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best_val': f'{best_val_loss:.4f}'
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"\n  🎉 New best model! Validation loss: {avg_val_loss:.4f}")
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            
            self.model.train()
        
        print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.4f}")
        print(f"📁 Model saved to {output_dir}")
        
        return None

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Calculate metrics
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir="./bert-base-ner-finetuned"):
        """Fine-tune the model"""
        print("Starting fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            push_to_hub=False
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            label_pad_token_id=-100
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer
    
    def evaluate_model(self, test_dataset, trainer=None):
        """Evaluate the model on test set"""
        print("Evaluating model...")
        
        if trainer is None:
            # If no trainer provided, create a simple evaluation
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            self.model.eval()
            
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert to format expected by seqeval
            true_predictions = [
                [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(all_predictions, all_labels)
            ]
            true_labels = [
                [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(all_predictions, all_labels)
            ]
        else:
            # Use trainer's evaluate method
            eval_results = trainer.evaluate(test_dataset)
            return eval_results
        
        # Generate classification report
        report = classification_report(true_labels, true_predictions, digits=4)
        print("\nClassification Report:")
        print(report)
        
        # Calculate overall metrics
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        
        print(f"\nOverall Metrics:")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'report': report
        }
    
    def predict(self, text):
        """Predict entities in new text"""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )
        
        # Extract offset_mapping before removing it
        offset_mapping = inputs['offset_mapping'][0].numpy()
        
        # Remove offset_mapping from inputs (model doesn't need it)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'offset_mapping'}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        # Convert predictions to labels
        predicted_labels = [self.id2label[pred] for pred in predictions]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (label, (start, end)) in enumerate(zip(predicted_labels, offset_mapping)):
            if start == 0 and end == 0:  # Skip special tokens
                continue
                
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': text[start:end],
                    'label': label[2:],
                    'start': start,
                    'end': end
                }
            elif label.startswith('I-') and current_entity:
                # Continuation of entity
                current_entity['text'] = text[current_entity['start']:end]
                current_entity['end'] = end
            else:
                # End of entity or no entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

def main():
    """Main execution function"""
    print("=== BERT NER Fine-tuning Pipeline ===\n")
    
    # Load your data
    print("Loading data...")
    df = pd.read_csv("cecilia100_fixed_final.csv")
    
    # Rename the annotation column if needed
    if len(df.columns) > 3:
        df.rename(columns={df.columns[3]: "INFO"}, inplace=True)
    
    print(f"Loaded {len(df)} rows")
    
    # Initialize fine-tuner
    fine_tuner = NERFineTuner()
    
    # Prepare data
    df_prepared = fine_tuner.prepare_data(df)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = fine_tuner.create_datasets(df_prepared)
    
    # Initialize model
    fine_tuner.initialize_model()
    
    # Train model
    trainer = fine_tuner.train_manual(train_dataset, val_dataset)
    
    # Evaluate model
    results = fine_tuner.evaluate_model(test_dataset, trainer)
    
    print("\n=== Fine-tuning Complete ===")
    print("Model saved to ./bert-base-ner-finetuned")
    
    # Example prediction
    sample_text = "The Taliban said their phones, email and website had been hacked to spread a false report that the movement's spiritual leader, Mullah Omar, was dead."
    print(f"\nExample prediction:")
    print(f"Text: {sample_text}")
    entities = fine_tuner.predict(sample_text)
    print("Entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['label']}) [{entity['start']}:{entity['end']}]")

if __name__ == "__main__":
    main()

# After restarting kernel, run this to check:
import transformers
import accelerate
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")

# Check if accelerate is properly detected
from transformers.utils import is_accelerate_available
print(f"Is accelerate available to transformers: {is_accelerate_available()}")

# First, manually remove the corrupted accelerate files

# Now install fresh
get_ipython().system('pip install accelerate>=0.26.0')


# Check if accelerate installed properly
import accelerate
print(f"Accelerate version: {accelerate.__version__}")

# Check if transformers can now detect it
from transformers.utils import is_accelerate_available
print(f"Is accelerate available to transformers: {is_accelerate_available()}")


# Let's debug the transformers detection function
from transformers.utils import is_accelerate_available
import importlib

# Check the actual detection logic
def debug_accelerate_detection():
    try:
        import accelerate
        print(f"✓ Accelerate imports successfully: {accelerate.__version__}")
        
        # Check if it has the required attributes
        required_attrs = ['__version__']
        for attr in required_attrs:
            if hasattr(accelerate, attr):
                print(f"✓ Has {attr}: {getattr(accelerate, attr)}")
            else:
                print(f"✗ Missing {attr}")
                
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

result = debug_accelerate_detection()
print(f"Manual detection result: {result}")
print(f"Transformers detection result: {is_accelerate_available()}")


# More aggressive monkey patching
import transformers.utils
import transformers.training_args

# Patch multiple places where the check might occur
def fixed_accelerate_check():
    return True

# Patch all the places this check might be used
transformers.utils.is_accelerate_available = fixed_accelerate_check
transformers.training_args.is_accelerate_available = fixed_accelerate_check

# Also patch the module-level import
import sys
if 'transformers.utils' in sys.modules:
    sys.modules['transformers.utils'].is_accelerate_available = fixed_accelerate_check

# Verify
from transformers.utils import is_accelerate_available
print(f"Fixed detection result: {is_accelerate_available()}")

