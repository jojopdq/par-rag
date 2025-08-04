import hashlib
import json
import random
import uuid
from typing import Dict

from qdrant_client import QdrantClient
from qdrant_client.models import MatchValue, VectorParams, Distance, PointStruct
from qdrant_client.models import Range, FieldCondition, Filter
from sentence_transformers import SentenceTransformer

from src.bert_classifier_plus import QuestionComplexityClassifier


class Director:
    def __init__(self, config: Dict, classifier: QuestionComplexityClassifier):
        self.config = config.get("store", {})
        self.encoder = SentenceTransformer(self.config["embedding_model_name"])
        self.client = QdrantClient(url=self.config["address"])
        self.collection_name = self.config["collection_name"]
        has_exist = self.client.collection_exists(self.collection_name)
        if not has_exist:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                    distance=Distance.COSINE,
                ),
            )
        self.classifier = classifier

    def save(self, question, plan):
        plan = json.dumps(plan)
        complexity_score = self.classifier.predict(question)
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=str(uuid.uuid3(uuid.NAMESPACE_DNS, question)),
                            vector=self.encoder.encode(question).tolist(),
                            payload={"question": question, "plan": plan, "complexity_score": complexity_score}),
            ],
        )
        return operation_info

    def fetch_most_similar_history_record(self, question):
        query_vector = self.encoder.encode(question)
        complexity_score = self.classifier.predict(question)
        query_filter = Filter(
            should=[
                FieldCondition(
                    key="complexity_score",
                    match=MatchValue(value=complexity_score),
                )
            ]
        )
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=5
        ).points
        result = None
        for hit in hits:
            result = {"question": hit.payload["question"], "plan": hit.payload["plan"]}
            break
        return complexity_score, result

    def fetch_random_history_record(self, question):
        query_vector = self.encoder.encode(question)
        complexity_score = self.classifier.predict(question)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            with_payload=True,
            limit=10
        ).points
        result = None
        random.shuffle(hits)
        for hit in hits:
            result = {"question": hit.payload["question"], "plan": hit.payload["plan"]}
            break
        return complexity_score, result
