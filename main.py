import argparse
import os
import sys

import pandas as pd
import yaml
from loguru import logger

from src.langchain_util import init_langchain_model
from src.par.commander import Commander
from src.par.evaluator import Evaluator
from src.retriever import Colbertv2Retriever

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", level="DEBUG")
logger.add("/tmp/par-rag.log", rotation="500 MB", level="INFO")


def benchmark(client, env, scope, round_name):
    if scope == 'all':
        baselines = ['standard-rag', 'raptor', 'hippo-rag', 'par-rag', 'react', 'self-ask', 'ircot']
    else:
        baselines = [scope]
    metric_file = f"output/{scope}_{round_name}.csv"
    if os.path.exists(metric_file):
        df = pd.read_csv(metric_file)
        print(df)
        return
    baseline_datasets = {}
    for baseline in baselines:
        root_path = f"output/{baseline}/{round_name}"
        if not os.path.exists(root_path):
            continue
        datasets = []
        for (dirpath, _, filenames) in os.walk(root_path):
            for filename in filenames:
                datasets.append(filename.split("_")[0])
        baseline_datasets[baseline] = datasets
    results = []
    for baseline in baselines:
        datasets = baseline_datasets.get(baseline, None)
        if datasets is None:
            continue
        evaluator = Evaluator(client, env, baseline, round_name)
        for dataset in datasets:
            metrics = evaluator.process(dataset)
            results.append({'dataset': dataset, 'method': baseline,
                            'em': metrics.get('em'),
                            'f1': metrics.get('f1'),
                            'acc': metrics.get('acc'),
                            })
    df = pd.DataFrame(results)
    df.to_csv(metric_file, index=False, encoding='utf-8')
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, default='run')
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa')
    parser.add_argument('--llm_provider', type=str, default='openai')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--env', type=str, default='dev')
    parser.add_argument('--ablation_mode', type=str, default=None)
    parser.add_argument('--example_mode', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--round_name', type=str, default=None)
    parser.add_argument('--scope', type=str, default='par-rag')
    args = parser.parse_args()
    logger.debug(f"args: {args}")

    corpus_name = args.dataset
    command = args.command
    env = args.env

    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    if not config:
        raise FileNotFoundError("config.yaml not found")

    dataset_path = config.get('data', {}).get('dataset_path')
    assert dataset_path is not None, "dataset_path not found"
    os.environ['QA_DATASET_DIR'] = dataset_path

    config["current_llm_provider"] = args.llm_provider
    config["current_llm_model"] = args.llm_model
    config["env"] = env
    config["corpus_name"] = corpus_name

    if args.command == 'dry-run' or args.command == 'run':

        config["ablation_mode"] = args.ablation_mode
        config["example_mode"] = args.example_mode
        config["round_name"] = args.round_name

        logger.debug(f"config: {config}")
        commander = Commander(config)
        if args.command == 'dry-run' and args.question is not None:
            query = args.question
            commander.execute_only_one(corpus_name, query)
        elif args.command == 'run':
            if args.round_name is None:
                raise ValueError("round_name must be specified")
            commander.execute(corpus_name)
    elif args.command == 'benchmark':
        client = init_langchain_model(llm=args.llm_provider, model_name=args.llm_model, config=config)
        benchmark(client, env, args.scope, args.round_name)
    elif args.command == 'build-index':
        retriever = Colbertv2Retriever(config)
        retriever.build_index()
    else:
        raise ValueError(f"Unknown command: {args.command}")
