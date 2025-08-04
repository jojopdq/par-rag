import os
from abc import abstractmethod
from typing import Dict

from colbert.infra import Run
from colbert.infra import RunConfig
from colbert import Searcher, Indexer
from colbert.infra import ColBERTConfig

from src.lib import get_qa_dataset_path, read_jsonl

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

TOP_K = 10


class DocumentRetriever:
    @abstractmethod
    def rank_docs(self, query: str, top_k: int):
        """
        Rank the documents in the corpus based on the given query
        :param query:
        :param top_k:
        :return: ranks and scores of the retrieved documents
        """


class Colbertv2Retriever(DocumentRetriever):
    def __init__(self, config: Dict):
        self.env = config['env']
        self.corpus_name = config['corpus_name']
        root_path = config['data']['index_path']
        self.index_name = self.corpus_name + '_nbits_2_' + self.env
        self.index_data_path = f"{root_path}/index_data"
        self.checkpoint_path = f"{root_path}/colbertv2.0"
        if os.path.exists(self.index_data_path):
            with Run().context(RunConfig(nranks=1, experiment=self.corpus_name, root=self.index_data_path)):
                config = ColBERTConfig(
                )
                self.searcher = Searcher(index=self.index_name, config=config)

    def rank_docs(self, query: str, top_k: int):
        results = []
        items = self.searcher.search(query, k=top_k)
        for passage_id, passage_rank, passage_score in zip(*items):
            results.append({'id': passage_id, 'score': passage_score, 'rank': passage_rank,
                            'content': self.searcher.collection[passage_id]})
        return results

    def fetch(self, question, top_k: int = 5):
        results = self.rank_docs(question, top_k=TOP_K)
        temp = [item.get('content') for item in results]
        temp = temp[0:top_k]
        return temp

    def run_colbertv2_index(self, corpus_tsv_path: str, overwrite=True):
        with Run().context(RunConfig(nranks=1, experiment=self.corpus_name, root=self.index_data_path)):
            config = ColBERTConfig(
                nbits=2,
            )
            indexer = Indexer(checkpoint=self.checkpoint_path, config=config)
            indexer.index(name=self.index_name, collection=corpus_tsv_path, overwrite=overwrite)
            print(f'Indexing done for dataset {self.corpus_name}, index {self.index_name}')

    def build_index(self):
        assert os.path.isdir(self.checkpoint_path)
        if not os.path.isdir(self.index_data_path):
            os.makedirs(self.index_data_path)
        corpus_path = os.path.join(f"{get_qa_dataset_path()}/processed_data", self.corpus_name,
                                   f"{self.env}_subsampled.jsonl")
        corpus = read_jsonl(corpus_path)
        corpus_contents = []
        for item in corpus:
            contexts = item.get("contexts", [])
            corpus_contents.extend([ctx['title'] + '\t' + ctx['paragraph_text'].replace('\n', ' ') for ctx in contexts])
        print('corpus len', len(corpus_contents))

        corpus_tsv_path = f'{self.index_data_path}/{self.corpus_name}_colbert_{self.env}.tsv'
        with open(corpus_tsv_path, 'w') as f:
            for pid, p in enumerate(corpus_contents):
                f.write(f"{pid}\t\"{p}\"" + '\n')
        print(f'Corpus tsv saved: {corpus_tsv_path}', len(corpus_contents))

        self.run_colbertv2_index(corpus_tsv_path, overwrite=True)
