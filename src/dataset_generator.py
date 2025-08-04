import os
import random
import sys
from collections import defaultdict, Counter

import numpy as np
import spacy
import torch
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.append('.')
from src.lib import read_jsonl

nlp = spacy.load("en_core_web_sm")
max_per_class = 1171
structure_scale = 2
random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
max_classes = 4
# embed_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name).to(device).eval()
test_dataset_file = 'data/test_dataset.npz'


def extract_hop(qid):
    prefix = qid.split("hop")[0]
    hop = int(prefix.replace("hop", "").strip())
    return hop - 1  # 0-indexed label


def estimate_entropy_kde(embedding, noise_scale=0.01, samples=100):
    perturbed = embedding + noise_scale * np.random.randn(samples, embedding.shape[0])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(perturbed)
    log_probs = kde.score_samples(perturbed)
    entropy = -np.mean(log_probs)
    return entropy


def dependency_tree_depth(text):
    doc = nlp(text)
    return max([len(list(tok.ancestors)) for tok in doc if tok.head != tok], default=0)


def estimate_complexity(question, emb=None):
    with torch.no_grad():
        if emb is None:
            inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
            emb = embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()
        doc = nlp(str(question))
        kde_entropy = torch.tensor(estimate_entropy_kde(emb)).cpu()
        # Tree depth
        depth = max([len(list(tok.ancestors)) for tok in doc])
        tree_depth = torch.tensor([depth]).cpu()

        # Extra features
        q_len = torch.tensor([len(doc)], dtype=torch.float32)
        puncts = torch.tensor([sum(1 for tok in doc if tok.is_punct)], dtype=torch.float32)
        ents = torch.tensor([len(doc.ents)], dtype=torch.float32)
        wh_count = torch.tensor([sum(1 for tok in doc if tok.lower_ in {"what", "where", "when", "who", "why", "how"})],
                                dtype=torch.float32)

        # Token-level variance entropy
        var = torch.var(torch.tensor([t.vector for t in doc if t.has_vector], dtype=torch.float32), dim=0)
        token_entropy = torch.tensor([var.mean().item()], dtype=torch.float32)

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)

        full = torch.cat([emb,
                          torch.tensor([structure_scale * kde_entropy.unsqueeze(0)], dtype=torch.float32),
                          torch.tensor([structure_scale * tree_depth], dtype=torch.float32),
                          q_len,
                          puncts,
                          ents,
                          wh_count,
                          token_entropy
                          ])

    return full, emb


def create_single_hop_dataset(split_mode, split_num):
    dataset = load_dataset("trivia_qa", "rc", split=f"{split_mode}[:{split_num}]")
    result = []
    for sample in dataset:
        sample["id"] = f"1hop_{sample['question_id']}"
        result.append({"id": sample["id"], "question": sample["question"]})
    random.shuffle(result)
    return Dataset.from_list(result)


def balanced_dataset(dataset, split_num):
    hop_buckets = defaultdict(list)

    for sample in dataset:
        hop = extract_hop(sample["id"])
        hop_buckets[hop].append({"id": sample["id"], "question": sample["question"]})

    balanced_samples = []
    for hop, samples in hop_buckets.items():
        if len(samples) >= split_num:
            selected = random.sample(samples, split_num)
        else:
            selected = samples
        balanced_samples.extend(selected)

    random.shuffle(balanced_samples)
    return Dataset.from_list(balanced_samples)


def generate_dataset_file(split_mode="train", split_num=500, d_mode="simple"):
    if d_mode == "complex":
        output_file = f'data/{d_mode}_{split_mode}_dataset_{split_num}_{structure_scale}.npz'
    else:
        output_file = f'data/{d_mode}_{split_mode}_dataset_{split_num}.npz'
    if os.path.exists(output_file):
        return
    single_hop_dataset = create_single_hop_dataset(split_mode, split_num)
    dataset = load_dataset("bdsaglam/musique", split=split_mode)
    dataset = balanced_dataset(dataset, split_num)
    dataset = concatenate_datasets([single_hop_dataset, dataset])
    dataset = dataset.shuffle()
    questions = dataset["question"]
    question_ids = dataset["id"]
    hop_labels = [extract_hop(qid) for qid in question_ids]
    print("Label distribution:", Counter(hop_labels))

    embeddings = []
    with torch.no_grad():
        for question in tqdm(questions, desc="Encoding questions"):
            inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = embed_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
            full, emb = estimate_complexity(question, outputs.cpu())
            if d_mode == "complex":
                embeddings.append(full.numpy())
            else:
                embeddings.append(emb.numpy())

    embeddings = np.stack(embeddings)
    hop_labels = np.array(hop_labels)

    np.savez_compressed(output_file, embeddings=embeddings, labels=hop_labels, questions=questions)
    print(f"saved: {output_file} (total {len(embeddings)})")


def generate_test_data_file():
    qa_data_dir = os.environ.get("QA_DATASET_DIR")
    if os.path.exists(test_dataset_file):
        return
    lines = read_jsonl(f"{qa_data_dir}/raw_data/musique/musique_full_v1.0_test.jsonl")
    samples = []
    for line in lines:
        samples.append({'question': line['question'], 'hop': extract_hop(line['id'])})
    another_lines = read_jsonl(f"{qa_data_dir}/raw_data/squad/dev_subsampled.jsonl")
    for line in another_lines:
        samples.append({'question': line['question_text'], 'hop': 0})
    another_lines = read_jsonl(f"{qa_data_dir}/raw_data/squad/test_subsampled.jsonl")
    for line in another_lines:
        samples.append({'question': line['question_text'], 'hop': 0})

    hop_buckets = defaultdict(list)

    for sample in samples:
        hop_buckets[sample['hop']].append(sample)

    min_size = len(hop_buckets[0])
    for hop, samples in hop_buckets.items():
        if len(samples) <= min_size:
            min_size = len(samples)
    balanced_samples = []
    for hop, samples in hop_buckets.items():
        if len(samples) >= min_size:
            selected = random.sample(samples, min_size)
        else:
            selected = samples
        balanced_samples.extend(selected)

    random.shuffle(balanced_samples)
    test_samples = balanced_samples
    test_labels = []
    test_questions = []
    for sample in test_samples:
        test_labels.append(sample['hop'])
        test_questions.append(sample['question'])

    test_labels = np.array(test_labels)
    test_questions = np.array(test_questions)
    print(f"Label Distribution: {Counter(test_labels)}, total samples: {len(test_samples)}")
    np.savez_compressed(test_dataset_file, labels=test_labels, questions=test_questions)
    # write_jsonl(test_samples, test_dataset_file)


if __name__ == "__main__":
    #
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    if not config:
        raise FileNotFoundError("config.yaml not found")

    dataset_path = config["data"]["dataset_path"]
    os.environ["QA_DATASET_DIR"] = dataset_path
    generate_dataset_file("train", max_per_class, "simple")
    generate_dataset_file("train", max_per_class, "complex")
    generate_dataset_file("validation", d_mode="simple")
    generate_dataset_file("validation", d_mode="complex")
    generate_test_data_file()
