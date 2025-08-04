# PAR-RAG

## 1. Overview

This archive contains:

- The source code (`src/`) to reproduce our main experiments
- Scripts to run training and evaluation (`scripts/`)

# 2. Installation

## Init Project

```
conda create -n par-rag python=3.10.16
conda activate par-rag
```

## Install Dependencies

```
pip install -r requirements.txt
```

# 3. Quick Start

## Download Dataset

Download dataset according to the instruction of [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)

## Config

Change config.yaml according to your available resources and environment.
If you don't need a proxy to access OpenAI, please keep the value of proxy address empty

## Build Index

```
DATASET=musique make index
```

## Training a bert classifier

### Preparation for dataset

Execute the following command to generate the training dataset.

```angular2html
python src/dataset_generator.py
```

### Training

```angular2html
python src/bert_classifier_plus.py train
```

```angular2html
python src/bert_classifier_plus.py validate 
```

### Usage

```angular2html
model_path = YOUR_MODEL_PATH
classifier = QuestionComplexityClassifier(model_path)
complexity_score = classifier.predict(YOUR_INPUT_TEXT)
print(complexity_score)
```

# 4. Experiment

## Run Test

### For a specific dataset

```
DATASET=musique make run
```

### For all datasets

```
sh scripts/run.sh
```

### Run ablation study over multiple datasets

```angular2html
sh scripts/run_ablation_study.sh
```

### For a specific question

```
python main.py dry-run --dataset hotpotqa --question "North Midland Divisional Engineers took part in a battle during WWII that went on for how many weeks?"
python main.py dry-run --dataset 2wikimultihopqa --question "Who is Ermengarde Of Tuscany's paternal grandfather?"
```

## Evaluation

```
python main.py benchmark --round_name YOUR_ROUND_NAME 
```

# 5.Acknowledgment

Part of the code for evaluation is based on [HippoRAG.](https://github.com/OSU-NLP-Group/HippoRAG)
and [Adaptive RAG](https://github.com/starsuzi/Adaptive-RAG)
