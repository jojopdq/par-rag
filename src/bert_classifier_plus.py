import argparse
import math
import os
import pickle
from collections import Counter

import nltk
import numpy as np
import torch
from matplotlib import pyplot as plt
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, \
    Trainer, DataCollatorWithPadding, EarlyStoppingCallback, BertTokenizer, BertForSequenceClassification, \
    BertPreTrainedModel, BertModel
from transformers import logging

logging.set_verbosity_error()
# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

model_name = 'bert-base-uncased'
# model_path = "output/bert-classifier-plus"
train_dataset_file = f'data/complex_train_dataset_1171.npz'
validation_dataset_file = 'data/complex_validation_dataset_500.npz'
test_dataset_file = 'data/test_dataset.npz'
model_file_name = "bert-classifier.pth"
feature_file_name = "features.pickle"
# feature_file = os.path.join(model_path, 'feature_stats.pickle')

max_classes = 4
epoch = 5
learning_rate = 3.05e-5
dropout = 0.5
batch_size = 8  # 16
weight_decay = 0.01
max_length = 128
import spacy

nlp = spacy.load("en_core_web_sm")

# Init
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def entropy(prob_dist):
    return -sum(p * math.log(p + 1e-10) for p in prob_dist if p > 0)


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)


def sentence_word_sense_entropy(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    entropies = []
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        if wn_pos:
            synsets = wn.synsets(word, pos=wn_pos)
            num_senses = len(synsets)
            if num_senses > 1:
                probs = [1.0 / num_senses] * num_senses
                h = entropy(probs)
                entropies.append(h)
            elif num_senses == 1:
                entropies.append(0.0)
    if entropies:
        return sum(entropies) / len(entropies), max(entropies)
    else:
        return 0.0, 0.0


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # predict most probable class
    preds = np.argmax(logits, axis=1)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    # acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)
    return {"precision": precision, "recall": recall, "macro_f1": macro_f1, "accuracy": accuracy}


def compute(q: str, label, tokenizer, stats):
    doc = nlp(q)

    length = len(q.split())
    depth = dependency_tree_depth(q)
    ner_count = len(doc.ents)
    entropy, _ = sentence_word_sense_entropy(q)

    norm_len = normalize(length, *stats['length'])
    norm_depth = normalize(depth, *stats['depth'])
    norm_ner = normalize(ner_count, *stats['ner'])
    norm_entropy = normalize(entropy, *stats['entropy'])

    inputs = tokenizer(q, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
    if label is not None:
        label = torch.tensor(label, dtype=torch.long)
    else:
        label = None
    item = {
        'input_ids': inputs['input_ids'].squeeze(0),
        'attention_mask': inputs['attention_mask'].squeeze(0),
        'length': torch.tensor(norm_len, dtype=torch.float),
        'depth': torch.tensor(norm_depth, dtype=torch.float),
        'ner': torch.tensor(norm_ner, dtype=torch.float),
        'entropy': torch.tensor(norm_entropy, dtype=torch.float),
        'labels': label
    }
    return item


class QADataset(Dataset):
    def __init__(self, questions, labels, tokenizer, stats):
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.stats = stats  # dict: {feature: (min, max)}

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = str(self.questions[idx])
        item = compute(q, self.labels[idx], self.tokenizer, self.stats)
        return item


class BertClassifierWithFeature(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )
        self.init_weights()

    def forward(self, input_ids, attention_mask, length, depth, ner, entropy, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        extra_feats = torch.stack([length, depth, ner, entropy], dim=1)
        x = torch.cat([cls_embed, extra_feats], dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}


def model_init():
    # config = BertConfig.from_pretrained(model_name, num_labels=max_classes)
    return BertForSequenceClassification.from_pretrained(model_name, num_labels=max_classes)


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8)
    }


def train(model, tokenizer, train_loader, validation_loader, best_run=False, model_path=None):
    if model_path is None:
        raise Exception("No model path provided")
    training_args = TrainingArguments(
        output_dir=model_path,  # Directory for saving model checkpoints
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",
        learning_rate=learning_rate,  # Start with a small learning rate
        per_device_train_batch_size=batch_size,  # Batch size per GPU
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,  # Number of epochs
        weight_decay=weight_decay,  # Regularization
        save_total_limit=3,  # Limit checkpoints to save space
        load_best_model_at_end=True,  # Automatically load the best checkpoint
        lr_scheduler_type="linear",
        # metric_for_best_model="macro_f1",
        # greater_is_better=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if best_run:
        trainer = Trainer(
            # model=model,  # Pre-trained BERT model
            model_init=model_init,
            args=training_args,  # Training arguments
            train_dataset=train_loader,
            eval_dataset=validation_loader,
            tokenizer=tokenizer,
            data_collator=data_collator,  # Efficient batching
            compute_metrics=compute_metrics,  # Custom metric
        )
        best_run = trainer.hyperparameter_search(
            direction="maximize",  # minimize mse 或 maximize f1
            backend="optuna",
            hp_space=hp_space,
            n_trials=20,
        )
        print("Best run: ", best_run)
        training_args.learning_rate = best_run.hyperparameters["learning_rate"]
        training_args.weight_decay = best_run.hyperparameters["weight_decay"]
        training_args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]
        training_args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
    print(training_args)

    trainer = Trainer(
        model=model,  # Pre-trained BERT model
        # model_init=model_init,
        args=training_args,  # Training arguments
        train_dataset=train_loader,
        eval_dataset=validation_loader,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Efficient batching
        compute_metrics=compute_metrics,  # Custom metric
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    trainer.train()
    print("=======Evaluation Results======")
    trainer.evaluate()

    # trainer.save_model(model_path)
    # tokenizer.save_pretrained(model_path)
    feature_file = f"{model_path}/{feature_file_name}"
    if not os.path.exists(feature_file):
        with open(feature_file, 'ab') as file:
            pickle.dump(train_loader.stats, file)
    torch.save(model.state_dict(), f'{model_path}/{model_file_name}')
    return model, tokenizer


def load_model(model_path):
    # tokenizer = BertTokenizer.from_pretrained(model_path)
    # model = BertClassifierWithFeature.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifierWithFeature.from_pretrained(model_name, num_labels=max_classes)
    model.load_state_dict(torch.load(f"{model_path}/{model_file_name}"))
    model.eval()
    return model, tokenizer


def dependency_tree_depth(text):
    doc = nlp(text)
    return max([len(list(tok.ancestors)) for tok in doc if tok.head != tok], default=0)


def load_dataset(data_file, tokenizer):
    data = np.load(data_file)
    dataset = []
    questions = [question for question in data['questions']]
    labels = [label for label in data['labels']]
    print(
        f"length of texts: {len(dataset)}, length of labels: {len(labels)}, distribution of labels: {Counter(labels)}")

    # train_tokenized = tokenizer(questions, truncation=True, padding=True, max_length=256)
    lengths = np.array([len(str(q).split()) for q in questions])
    depths = np.array([dependency_tree_depth(str(q)) for q in questions])
    entropies = np.array([sentence_word_sense_entropy(str(q)) for q in questions])
    ner_counts = []
    for q in questions:
        doc = nlp(str(q))
        ner_counts.append(len(doc.ents))
    ner_counts = np.array(ner_counts)

    feature_stats = {
        'length': (lengths.min(), lengths.max()),
        'depth': (depths.min(), depths.max()),
        'ner': (ner_counts.min(), ner_counts.max()),
        'entropy': (entropies.min(), entropies.max()),
    }
    current_dataset = QADataset(questions, labels, tokenizer, feature_stats)
    return current_dataset


def test(model, tokenizer, data_file, feature_stats=None):
    test_dataset = load_dataset(data_file, tokenizer)
    if feature_stats:
        test_dataset.stats = feature_stats
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            length = batch['length']
            depth = batch['depth']
            ner = batch['ner']
            entropy = batch['entropy']

            labels = batch.get('labels', None)
            if labels is not None:
                labels = labels

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                length=length,
                depth=depth,
                ner=ner,
                entropy=entropy,
                labels=labels
            )

            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            if labels is not None:
                all_labels.extend(labels.cpu().tolist())

    if all_labels:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        acc = correct / len(all_labels)
        print(f"Test Accuracy: {acc:.4f}")
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=['1-hop', '2-hop', '3-hop', '4-hop']))
    from sklearn.metrics import f1_score
    print("Macro F1:", f1_score(all_labels, all_preds, average='macro'))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1-hop", "2-hop", "3-hop", "4-hop"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


class QuestionComplexityClassifier:
    def __init__(self, model_path):
        model, tokenizer = load_model(model_path)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.feature_path = f"{model_path}/{feature_file_name}"
        with open(self.feature_path, 'rb') as file:
            self.feature_stats = pickle.load(file)

    def predict(self, text):
        doc = nlp(text)

        length = len(text.split())
        depth = dependency_tree_depth(text)
        ner_count = len(doc.ents)
        entropy, _ = sentence_word_sense_entropy(text)

        norm_len = normalize(length, *self.feature_stats['length'])
        norm_depth = normalize(depth, *self.feature_stats['depth'])
        norm_ner = normalize(ner_count, *self.feature_stats['ner'])
        norm_entropy = normalize(entropy, *self.feature_stats['entropy'])

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        length = torch.tensor([norm_len], dtype=torch.float)
        depth = torch.tensor([norm_depth], dtype=torch.float)
        ner = torch.tensor([norm_ner], dtype=torch.float)
        entropy = torch.tensor([norm_entropy], dtype=torch.float)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            length=length,
            depth=depth,
            ner=ner,
            entropy=entropy
        )

        probs = torch.softmax(outputs['logits'], dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        return pred + 1


def validate(model_path):
    model, tokenizer = load_model(model_path)
    model.eval()
    feature_path = f"{model_path}/{feature_file_name}"
    with open(feature_path, 'rb') as file:
        feature_stats = pickle.load(file)
    print(f"===========Test===========")
    test(model, tokenizer, validation_dataset_file, feature_stats)


def simple_test(model_path):
    classifier = QuestionComplexityClassifier(model_path)

    trajectories = [
        ("Where did the wrestler who has held the intercontinental championship the most times win in 2008?", 2),
        ("Who did the Huguenots in the state primary Edwards won outside of Georgia buy land from?", 2),
        (
            "When did the party that is the current majority in the political assembly that passed a bill giving regulators discretion to prohibit proprietary trades, gain control of the House?",
            3),
        ("who was president when france gave us the statue of liberty", 1),
        (
            "What date did the explorer reach the location of the headquarters of the only company larger than BMG's partner from 2005-2007?",
            4),
        ("Who sings home alone tonight with the singer of you can crash my party anytime?", 2),
        ("The capital of the state where Mary Boykin Chesnut lived borders a city within which county?", 4),
        ("What is the name of the famous bridge in the birthplace of Bajazet's composer?", 3),
        (
            "When was the last time the winner of the American League East in 2017 met the Dodgers in the event after which the MLB MVP award is given out?",
            3),
        ("who wrote the song paint me a birmingham", 1),
        (
            "Who burned down the city where Dunn Dunn's recording artist died during the conflict after which occurred the historical period of A Rose for Emily?",
            4),
        ("when did mandela write long walk to freedom", 1),
        ("When did the president of Notre Dame in 2012 begin his tenure?", 2),
        ("What is the seat of the county sharing a border with the county where Don Werner was born?", 4),
        ("What is the seat of the county sharing a border with the county in which Miller Electric is headquartered?",
         4),
        ("when did the movie love and basketball come out", 1),
        ("Who wrote the TV Series containing the Finale episodes?", 2),
        (
            "Where does the river by the city sharing a border with Elizabeth Berg's birthplace empty into the Gulf of Mexico?",
            4),
    ]
    err_count = 0
    for trajectory in trajectories:
        text, label = trajectory
        pred = classifier.predict(text)
        if pred != label:
            err_count += 1
        print(f"Trajectory: {text}\nlabel: {label}, pred: {pred}")
    print(f"Correction Rate: {(len(trajectories) - err_count) / len(trajectories)}")


if __name__ == '__main__':
    model_path = "data/model"
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, default='run')
    args = parser.parse_args()

    command = args.command
    if command == 'train':
        # Load or create model
        if not os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertClassifierWithFeature.from_pretrained(model_name, num_labels=max_classes)
            # Create dataset and loader for each category
            train_dataset = load_dataset(train_dataset_file, tokenizer)
            validation_dataset = load_dataset(validation_dataset_file, tokenizer)
            train(model, tokenizer, train_dataset, validation_dataset, model_path=model_path)
    elif command == 'validate':
        validate(model_path)
    else:
        simple_test(model_path)
