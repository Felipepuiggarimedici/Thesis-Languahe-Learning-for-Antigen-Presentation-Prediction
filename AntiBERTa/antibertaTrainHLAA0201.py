from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    
)
import re
import json
#so progress can be seen
from tqdm.notebook import tqdm
import transformers.trainer_utils as trainer_utils

trainer_utils.tqdm = tqdm
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import pandas as pd
import torch
import numpy as np
import random
import os
#very important for running in HPC
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Initialise the tokeniser
tokenizer = RobertaTokenizer.from_pretrained(
    "tokenizer"
)

# Initialise the data collator, which is necessary for batching
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

print(f"Number of GPUs available: {torch.cuda.device_count()}")

def set_seed(seed: int = 42):
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()
hla_df = pd.read_csv("data/fullData/common_hla_sequence.csv")

from datasets import load_dataset

full_df = pd.read_csv('data/fullData/train_HLA_A0201.csv')
hlaName = "HLA-A:02-01"

print("Setting up HLA:", hlaName)
# Load filtered dataset
dataset = load_dataset('csv', data_files={'full': 'data/fullData/train_HLA_A0201.csv'})['full']

# Split dataset into train and eval (e.g., 90% train, 10% eval)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

# Tokenize train dataset
tokenized_train = train_dataset.map(
    lambda z: tokenizer(
        z["peptide"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    batched=True,
    num_proc=1,
    remove_columns=["peptide"],
)
# Tokenize eval dataset
tokenized_eval = eval_dataset.map(
    lambda z: tokenizer(
        z["peptide"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    batched=True,
    num_proc=1,
    remove_columns=["peptide"],
)

# These are the cofigurations they used for pre-training.
antiberta_config = {
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 26,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 16,
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}
# Initialise the model
model_config = RobertaConfig(
    vocab_size=antiberta_config.get("vocab_size"),
    hidden_size=antiberta_config.get("hidden_size"),
    max_position_embeddings=antiberta_config.get("max_position_embeddings"),
    num_hidden_layers=antiberta_config.get("num_hidden_layers", 12),
    num_attention_heads=antiberta_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)
    
# construct training arguments
# Huggingface uses a default seed of 42
args = TrainingArguments(
    output_dir="test",
    overwrite_output_dir=True,
    per_device_train_batch_size=antiberta_config.get("batch_size", 16),
    per_device_eval_batch_size=antiberta_config.get("batch_size", 16),
    max_steps=150000,
    save_steps=2500,
    logging_steps=100,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    learning_rate=1e-5,
    gradient_accumulation_steps=antiberta_config.get("gradient_accumulation_steps", 1),
    disable_tqdm=False,
    fp16=True,
    eval_strategy="steps",
     remove_unused_columns=True, 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,        
    save_total_limit=1,  
)

early_stopping_callback = EarlyStoppingCallback(    
    early_stopping_patience=10, 
    early_stopping_threshold=0.001
)
class LossLoggerCallback(TrainerCallback):
    def __init__(self, logIntervalSteps=10, printIntervalSteps=100, saveAfterStep=10):
        self.trainLosses = []
        self.evalLosses = []
        self.loggingSteps = []
        self.logIntervalSteps = logIntervalSteps
        self.printIntervalSteps = printIntervalSteps
        self.saveAfterStep = saveAfterStep

    def on_log(self, args, state, control, logs=None, **kwargs):
        currentStep = state.global_step
        if logs is not None and currentStep >= self.saveAfterStep:
            # Logic to STORE every 'logIntervalSteps'
            if currentStep % self.logIntervalSteps == 0:
                if 'loss' in logs:
                    self.trainLosses.append(logs['loss'])
                else:
                    self.trainLosses.append(None)

                if 'eval_loss' in logs:
                    self.evalLosses.append(logs['eval_loss'])
                else:
                    self.evalLosses.append(None)
                self.loggingSteps.append(currentStep)

            # Logic to PRINT only every 'printIntervalSteps'
            if currentStep % self.printIntervalSteps == 0:
                print(f"Step: {currentStep}, Train Loss: {logs.get('loss', float('nan')):.4f}, Eval Loss: {logs.get('eval_loss', float('nan')):.4f}")

loss_logger_callback = LossLoggerCallback() 
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset= tokenized_train,
    eval_dataset = tokenized_eval,
    callbacks=[early_stopping_callback, loss_logger_callback],
)
trainer.train()

safe_hla_name = re.sub(r'[^A-Za-z0-9_\-]', '_', hlaName)
model_dir = os.path.join("modelsPeptideOnly", f"HLA_{safe_hla_name}")
os.makedirs(model_dir, exist_ok=True)
trainer.save_model(model_dir)
print(f"Training for {hlaName} finished. Model saved to {model_dir}.")

loss_data = {
    'step': loss_logger_callback.loggingSteps,
    'train_loss': loss_logger_callback.trainLosses,
    'eval_loss': loss_logger_callback.evalLosses
}
loss_df = pd.DataFrame(loss_data)
loss_csv_path = os.path.join(model_dir, f"loss_history_{safe_hla_name}.csv")
loss_df.to_csv(loss_csv_path, index=False)
print(f"Loss history for {hlaName} saved to {loss_csv_path}.")

loss_json_path = os.path.join(model_dir, f"loss_history_{safe_hla_name}.json")
with open(loss_json_path, 'w') as f:
    json.dump(loss_data, f, indent=4)
print(f"Loss history for {hlaName} saved to {loss_json_path}.")