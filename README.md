# ML Customer Support Assistant

## Описание проекта

Прототип интеллектуального ассистента для обработки запросов сотрудников и клиентов.

Система объединяет:

- классификацию намерений пользователя;
- semantic retrieval;
- selective classification с reject threshold;
- маршрутизацию через LangGraph;
- Document RAG для ответов по внутренним документам;
- передачу недостаточно уверенных запросов на ручную проверку.

Основная задача — не только получить высокую точность, но и ограничить автоматические ответы в случаях, когда уверенность системы недостаточна.

---

# 1. Данные

Для классификации использовался датасет Banking77.

Количество intent-категорий: **77**.

Разделение данных:

- train_part: 8494 примеров;
- val_part: 1499 примеров;
- test: 3076 примеров.

### Назначение выборок

**train_part**

Используется для обучения моделей и построения retrieval-корпуса.

**val_part**

Используется для выбора confidence threshold и архитектуры системы.

**test**

Используется только для однократной финальной оценки после фиксации всех решений.

---

# 2. DistilBERT

Базовая модель:

`distilbert-base-uncased`

Количество эпох: **6**

Лучший checkpoint: `checkpoint-1330`

### Validation

- Accuracy: **90.79%**
- Macro F1: **90.55%**

### Final test

- Accuracy: **92.56%**
- Macro F1: **92.55%**
- Weighted F1: **92.55%**

---

# 3. Semantic Retrieval

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

Retrieval corpus построен только на `train_part`.

### Validation

- Hit@1: **90.26%**
- Hit@3: **95.26%**
- Hit@5: **97.00%**

### Final test

- Hit@1: **91.42%**
- Hit@3: **95.48%**
- Hit@5: **96.72%**

---

# 4. Selective Classification

Для отказа от недостаточно уверенных автоматических ответов используется confidence threshold.

Финальный threshold: **0.65**

Threshold был выбран исключительно на validation выборке.

### Final test

- Coverage: **93.95%**
- Automatic accuracy: **95.22%**
- Human review rate: **6.05%**
- Auto-closed: **2890**
- Automatic errors: **138**

---

# 5. LangGraph

В проекте проверена архитектура selective consensus.

Логика эксперимента:

1. При confidence DistilBERT выше threshold запрос закрывается автоматически.
2. Для low-confidence запросов запускается semantic retrieval.
3. При согласии DistilBERT и retrieval проверяется retrieval similarity.
4. При отсутствии достаточной уверенности запрос передаётся человеку.

Финальный retrieval similarity gate: **1.0**

### Результат эксперимента

При требовании automatic accuracy не ниже 95% дополнительная consensus-ветка не смогла безопасно увеличить coverage.

Поэтому validation-optimal frozen-конфигурация LangGraph фактически совпала с baseline `DistilBERT + reject threshold`.

### Final test

- Coverage: **93.95%**
- Automatic accuracy: **95.22%**
- Human review: **6.05%**
- Extra closed vs baseline: **0**

---

# 6. Document RAG

Document RAG предназначен для ответа на вопросы по внутренним документам и инструкциям.

### Evaluation dataset

Для финальной оценки использовано **30** QA-примеров.

Вопросы были сгенерированы через внешний API по случайно выбранным chunks.

Generator model: `openai/gpt-oss-120b`

Этот набор не использовался для настройки retrieval или RAG.

### Retrieval metrics

- Source accuracy: **96.67%**
- Chunk Hit@1: **66.67%**
- Chunk Hit@3: **93.33%**
- Runtime errors: **0**

### Answer quality

- Answer accuracy: **93.33%**
- Groundedness: **76.67%**
- Mean judge score: **3.70 / 4**
- Judge success rate: **100.00%**
- Overall RAG success: **76.67%**

---

# 7. Финальные результаты

| Компонент | Метрика | Результат |
|---|---|---:|
| DistilBERT | Accuracy | 92.56% |
| DistilBERT | Macro F1 | 92.55% |
| Semantic Retrieval | Hit@1 | 91.42% |
| Semantic Retrieval | Hit@3 | 95.48% |
| Semantic Retrieval | Hit@5 | 96.72% |
| Selective baseline | Coverage | 93.95% |
| Selective baseline | Automatic accuracy | 95.22% |
| Selective baseline | Human review | 6.05% |
| Document RAG | Source accuracy | 96.67% |
| Document RAG | Answer accuracy | 93.33% |
| Document RAG | Overall success | 76.67% |

---

# 8. Основные выводы

1. Финальный DistilBERT показал Accuracy 92.56% и Macro F1 92.55% на отложенной test выборке.

2. Confidence threshold позволяет увеличить надёжность автоматически закрываемых запросов до уровня выше 95%.

3. При threshold 0.65 автоматически обрабатывается 93.95% запросов.

4. Semantic retrieval показывает высокий Hit@3 и Hit@5 и может использоваться как вспомогательный компонент.

5. Эксперимент с LangGraph показал, что усложнение архитектуры само по себе не гарантирует улучшения.

6. При ограничении automatic accuracy >= 95% consensus-ветка LangGraph не дала безопасного прироста coverage.

7. Document RAG показал высокую точность ответов, однако groundedness и overall success остаются направлениями для дальнейшего улучшения.

8. Evaluation protocol разделяет обучение, выбор решений и финальную оценку, что снижает риск data leakage.

---

# 9. Структура проекта

```text
ml_customer_classification_corrected/
│
├── data/
│   ├── train_part.parquet
│   ├── val_part.parquet
│   ├── test.parquet
│   ├── label_mapping.json
│   └── label_mapping_final.json
│
├── models/
│   └── distilbert_final_6epoch/
│
├── validation/
│   ├── distilbert_final_val_predictions.csv
│   ├── distilbert_final_threshold_sweep.csv
│   └── selected_threshold_final.json
│
├── semantic/
│   └── results/
│
├── langgraph/
│   └── results/
│       ├── final_langgraph_architecture.json
│       ├── low_confidence_consensus_validation.csv
│       └── retrieval_similarity_threshold_sweep.csv
│
├── document_rag/
│   ├── data/
│   │   └── api_generated_qa.csv
│   └── results/
│       ├── api_generated_qa_rag_evaluation.csv
│       ├── api_generated_qa_rag_metrics.json
│       ├── api_generated_qa_final_evaluation.csv
│       └── api_generated_qa_final_metrics.json
│
├── results/
│   ├── FINAL_CORRECTED_SUMMARY.json
│   ├── final_test_metrics.json
│   ├── final_test_predictions.csv
│   ├── final_langgraph_test_predictions.csv
│   ├── report_classification_metrics.csv
│   ├── report_baseline_vs_langgraph.csv
│   ├── report_document_rag_metrics.csv
│   ├── report_evaluation_protocol.csv
│   └── DEFENSE_CONCLUSIONS.txt
│
└── README.md
```

---

# 10. Воспроизводимость

Seed: `42`

Все ключевые решения, включая confidence threshold и LangGraph architecture, были выбраны на validation.

Test использовался только после фиксации параметров.

После получения test-метрик архитектура и thresholds не изменялись.

---

<!-- GITHUB_MLSD_SECTION -->

# ML System Design и архитектура

Для проекта подготовлен отдельный ML System Design документ, описывающий архитектуру системы, data flow, ML-компоненты, deployment, monitoring, retraining strategy и основные риски.

### Основные документы

- [ML System Design](docs/mlsd.md)
- [High-Level Architecture](assets/architecture_high_level.md)
- [Intent Inference Flow](assets/intent_inference_flow.md)
- [Document RAG Flow](assets/document_rag_flow.md)
- [Training & Evaluation Flow](assets/training_evaluation_flow.md)

## Финальные результаты

| Метрика | Значение |
|---|---:|
| DistilBERT Test Accuracy | 92.56% |
| DistilBERT Test Macro F1 | 92.55% |
| Selective Coverage | 93.95% |
| Automatic Accuracy | 95.22% |
| Document RAG Answer Accuracy | 93.33% |
| Document RAG Overall Success | 76.67% |

## Структура репозитория

```text
ml_customer_support_assistant/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── config.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── routing.py
│   └── label_mapping.json
│
├── validation/
│   ├── selected_threshold_final.json
│   └── distilbert_final_threshold_sweep.csv
│
├── langgraph/
│   ├── final_langgraph_architecture.json
│   ├── low_confidence_consensus_validation.csv
│   └── retrieval_similarity_threshold_sweep.csv
│
├── document_rag/
│   ├── api_generated_qa.csv
│   ├── api_generated_qa_rag_metrics.json
│   ├── api_generated_qa_final_metrics.json
│   └── api_generated_qa_final_evaluation.csv
│
├── results/
│   ├── FINAL_CORRECTED_SUMMARY.json
│   ├── final_test_metrics.json
│   ├── report_classification_metrics.csv
│   ├── report_baseline_vs_langgraph.csv
│   ├── report_document_rag_metrics.csv
│   ├── report_evaluation_protocol.csv
│   └── DEFENSE_CONCLUSIONS.txt
│
├── docs/
│   └── mlsd.md
│
└── assets/
    ├── architecture_high_level.md
    ├── intent_inference_flow.md
    ├── document_rag_flow.md
    └── training_evaluation_flow.md
```

## Ограничения

Тяжёлые model checkpoints и parquet-датасеты не хранятся в GitHub-репозитории.

Финальная модель и исходные train / validation / test artifacts сохраняются отдельно, а репозиторий содержит код, конфигурацию и воспроизводимые результаты.

API keys и другие secrets в репозиторий не включаются.

---

<!-- FINAL_IMPLEMENTATION_SECTION -->

# Реализованные production-oriented компоненты

Помимо экспериментальной ML-части, в репозитории реализованы дополнительные компоненты system design.

### LangGraph orchestration

Реализован исполняемый end-to-end граф с маршрутизацией между intent classification, semantic retrieval, Document RAG, additional search и human review.

Код: `src/langgraph_app.py`

Smoke-test: `results/langgraph_smoke_test.json`

### Structured Output

Реализованы строгие Pydantic-схемы для intent results, RAG sources и финального ответа ассистента.

Код: `src/schemas.py`

Smoke-test: `results/structured_output_smoke_test.json`

### Tool Calling

Реализован внутренний tool registry с Pydantic-валидацией аргументов и безопасным executor.

Код: `src/tools.py`

Smoke-test: `results/tool_calling_smoke_test.json`

### Что остаётся production design

- отдельная production vector database (например, Qdrant);
- cross-encoder reranker;
- production monitoring / observability stack;
- интеграция tools с реальными CRM/API;
- production deployment infrastructure.
