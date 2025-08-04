SHELL=/bin/bash -o pipefail

DATASET ?= hotpotqa
LLM_PROVIDER ?= openai
MODEL ?= gpt-4o-mini
ENV ?= dev
ROUND_NAME ?= $(shell uuidgen)
EXAMPLE_MODE ?= similar

.PHONY: index
index:
	python main.py build-index --dataset ${DATASET} --env ${ENV} --llm_provider ${LLM_PROVIDER} --llm_model ${MODEL}

.PHONY: run
run:
	python main.py run --dataset ${DATASET}  --round_name ${ROUND_NAME} --example_mode ${EXAMPLE_MODE} --llm_provider ${LLM_PROVIDER} --llm_model ${MODEL} --env ${ENV}

.PHONY: train
train:
	python src/bert_classifier_plus.py train
