# 🌐 Multilingual Sentiment Analysis with LLaMA 3.1

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A parameter-efficient sentiment analysis system fine-tuned on **LLaMA 3.1 8B** to classify text sentiment across **13 Indian languages** using LoRA (Low-Rank Adaptation) and 4-bit quantization.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project fine-tunes the **LLaMA 3.1 8B Instruct** model for binary sentiment classification (Positive/Negative) across multiple Indian languages. By leveraging **LoRA** and **4-bit quantization**, the model achieves high accuracy while reducing trainable parameters to just **0.09%** of the total model size.

### Why This Project?
- **Multilingual NLP**: Supports 13 Indian languages with diverse scripts
- **Efficiency**: Uses parameter-efficient fine-tuning (PEFT) for resource-constrained environments
- **Production-Ready**: 4-bit quantization enables deployment on consumer GPUs
- **State-of-the-Art**: Leverages LLaMA 3.1's advanced language understanding

---

## ✨ Features

- 🚀 **Fine-tuned LLaMA 3.1 8B** with LoRA for sentiment classification
- 🌍 **13 Indian Languages**: Bengali, Gujarati, Tamil, Punjabi, Hindi, Marathi, Urdu, Kannada, Telugu, Odia, Malayalam, Assamese, Bodo
- ⚡ **4-bit Quantization**: Reduces memory footprint by 75% using BitsAndBytes
- 📊 **92.8% F1-Score**: High performance on multilingual test set
- 🔧 **Parameter-Efficient**: Only 6.8M trainable parameters (0.09% of total)
- 💾 **Memory Optimized**: Runs on single GPU with 16GB VRAM

---

## 📊 Dataset

### Dataset Statistics
- **Total Samples**: 1,000 labeled examples
- **Class Distribution**: 
  - Positive: 507 samples (50.7%)
  - Negative: 493 samples (50.3%)
- **Languages**: 13 Indian languages (~77 samples per language)
- **Train-Test Split**: 930 train / 70 test (7% test split)

### Supported Languages
| Language | Code | Script | Samples |
|----------|------|--------|---------|
| Bengali | bn | Bengali | 77 |
| Gujarati | gu | Gujarati | 77 |
| Tamil | ta | Tamil | 77 |
| Punjabi | pa | Gurmukhi | 77 |
| Hindi | hi | Devanagari | 77 |
| Marathi | mr | Devanagari | 77 |
| Urdu | ur | Perso-Arabic | 77 |
| Kannada | kn | Kannada | 77 |
| Telugu | te | Telugu | 77 |
| Odia | or | Odia | 77 |
| Malayalam | ml | Malayalam | 76 |
| Assamese | as | Bengali-Assamese | 77 |
| Bodo | bd | Devanagari | 77 |

---

## 🏗️ Model Architecture

### Base Model
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Architecture**: LlamaForSequenceClassification
- **Total Parameters**: 7.5B
- **Decoder Layers**: 32
- **Hidden Size**: 4096
- **Attention Heads**: 32

LoraConfig(
r=16, # Rank of LoRA matrices
lora_alpha=32, # Scaling factor
lora_dropout=0.05, # Dropout probability
target_modules=["q_proj", "v_proj"], # Applied to attention layers
task_type="SEQ_CLS", # Sequence classification
)


### Quantization
- **Method**: 4-bit NormalFloat (NF4)
- **Compute dtype**: bfloat16
- **Double quantization**: Disabled
- **Memory reduction**: ~75% compared to full precision

### Trainable Parameters
- **Trainable**: 6,823,936 (0.09%)
- **Total**: 7,511,756,800
- **Efficiency**: 1,100x reduction in trainable parameters

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- CUDA Toolkit 11.8+

### Setup

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


### Requirements
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
datasets>=2.14.0
evaluate>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0


---

## 💻 Usage

### 1. Data Preparation
from datasets import load_dataset

Load your dataset
dataset = load_dataset("csv", data_files={"train": "train.csv"})

Apply label mapping
label_map = {"Positive": 1, "Negative": 0}
dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})


### 2. Model Training
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

Load model with quantization
model = AutoModelForSequenceClassification.from_pretrained(
"meta-llama/Llama-3.1-8B-Instruct",
num_labels=2,
quantization_config=bnb_config,
device_map="auto"
)

Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

Train
trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=eval_dataset,
)
trainer.train()


### 3. Inference
from transformers import TextClassificationPipeline

Create pipeline
classifier = TextClassificationPipeline(
model=model,
tokenizer=tokenizer,
device="cuda"
)

Predict sentiment
text = "यह उत्पाद बहुत अच्छा है" # Hindi: "This product is very good"
prediction = classifier(text)
print(prediction)

Output: [{'label': 'Positive', 'score': 0.9206}]


### 4. Batch Prediction
Predict on multiple texts
test_texts = ["Sample text 1", "Sample text 2", ...]
predictions = classifier(test_texts, batch_size=32)


---

## 📈 Results

### Training Metrics
| Step | Training Loss | Validation Loss | F1-Score |
|------|---------------|-----------------|----------|
| 50   | 1.341         | 0.696           | 66.81%   |
| 100  | 0.423         | 0.310           | 91.40%   |
| 150  | 0.223         | 0.341           | 92.84%   |
| **200** | **0.251**     | **0.297**       | **92.86%** |

### Final Performance
- **F1-Score (Macro)**: 92.8%
- **Training Time**: ~1 hour on T4 GPU
- **Best Checkpoint**: Step 200
- **Convergence**: Achieved at epoch 2

### Performance Comparison
| Metric | Initial (Step 50) | Final (Step 200) | Improvement |
|--------|-------------------|------------------|-------------|
| F1-Score | 66.81% | 92.86% | +39.0% |
| Validation Loss | 0.696 | 0.297 | -57.3% |

---

## 🔧 Technical Details

### Hyperparameters
TrainingArguments(
num_train_epochs=2,
per_device_train_batch_size=4,
per_device_eval_batch_size=4,
gradient_accumulation_steps=2,
learning_rate=1e-4,
weight_decay=0.02,
adam_beta1=0.05,
adam_beta2=0.995,
fp16=True,
eval_steps=50,
save_steps=50,
logging_steps=50,
)


### Tokenization
- **Max Length**: 512 tokens
- **Padding**: Dynamic padding with DataCollatorWithPadding
- **Truncation**: Enabled
- **Special Tokens**: EOS token as padding token

### Evaluation Metric
- **Metric**: F1-Score (Macro-averaged)
- **Why Macro F1**: Accounts for class imbalance across languages

---

## 📁 Project Structure
multilingual-sentiment-analysis/
│
├── data/
│ ├── train.csv # Training dataset
│ ├── test.csv # Test dataset (unlabeled)
│ └── sample_submission.csv # Submission format
│
├── notebooks/
│ └── Sentiment_Analysis.ipynb # Main training notebook
│
├── models/
│ └── lora_llama8b_ct/ # Fine-tuned model checkpoints
│
├── logs/ # Training logs
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # MIT License


---

## 🛠️ Advanced Usage

### Custom Dataset Format
Your CSV should have these columns:
ID,sentence,label,language
1,"Sample text in Hindi",Positive,hi
2,"Sample text in Tamil",Negative,ta


### Model Export
Save fine-tuned model
model.save_pretrained("./my_sentiment_model")
tokenizer.save_pretrained("./my_sentiment_model")

Load later
from peft import PeftModel
base_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my_sentiment_model")


### Deployment
For production deployment
model = model.merge_and_unload() # Merge LoRA weights
model.save_pretrained("./production_model")


### LoRA Configuration
