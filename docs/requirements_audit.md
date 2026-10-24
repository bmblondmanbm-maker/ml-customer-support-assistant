# Audit of Original Project Requirements

Этот документ разделяет реально реализованные компоненты prototype и элементы, которые описаны только как production design.

| Компонент | Статус | Что есть сейчас | Production design |
|---|---|---|---|
| Python | IMPLEMENTED | Основной prototype, обучение, evaluation, retrieval и RAG реализованы на Python. | Python остаётся основным языком backend/ML service. |
| DistilBERT / Intent Classification | IMPLEMENTED | Обучен DistilBERT для 77 intents Banking77. Финальная test accuracy 92.56%, Macro F1 92.55%. | Используется как быстрый intent-classification service. |
| Embeddings | IMPLEMENTED | Использован sentence-transformers/all-MiniLM-L6-v2 для semantic retrieval и Document RAG. | Embedding service может быть выделен в отдельный production-компонент. |
| Semantic Retrieval | IMPLEMENTED | Построен retrieval corpus только по train_part. На final test Hit@1=91.42%, Hit@3=95.48%, Hit@5=96.72%. | Может использоваться для fallback и поиска похожих запросов. |
| RAG | IMPLEMENTED | Document RAG реализован и оценён на независимом API-generated QA benchmark. | Production-версия должна включать обновляемый vector index и document versioning. |
| LLM | IMPLEMENTED | Для Document RAG и offline evaluation использовался LLM через Groq API. | В production LLM должен использоваться только для генеративных маршрутов. |
| LangGraph | IMPLEMENTED | Реализован end-to-end LangGraph workflow: query type detection, intent classification, semantic retrieval, Document RAG, additional search, answer validation и human review. Smoke-test успешно прошёл все ключевые маршруты. | Production-версия может подключать реальные model services, vector database и monitoring через adapters. |
| Vector Database | PARTIAL | В prototype использован semantic vector search по embeddings. | Для production предлагается Qdrant. FAISS подходит для локального prototype. |
| Reranker | PROPOSED | Отдельный cross-encoder reranker в финальном prototype не реализован. | Предлагается добавить после initial retrieval для повышения качества Document RAG. |
| Structured Output | IMPLEMENTED | Реализованы строгие Pydantic-схемы для intent results, RAG sources и финального AssistantResponse. Проверены validation, invalid-output rejection и JSON serialization. | В production эти схемы могут использоваться как контракт API и LangGraph nodes. |
| Tool Calling | IMPLEMENTED | Реализован внутренний tool registry с Pydantic-валидацией аргументов и безопасным executor. Проверены корректный вызов, неизвестный tool, ошибочные аргументы и JSON serialization. | В production mock-tools заменяются реальными внутренними API/CRM/справочниками. |
| Human Review | IMPLEMENTED_AS_POLICY | Reject-policy рассчитывает долю запросов, которые должны быть переданы человеку. | Для production нужен отдельный интерфейс оператора и feedback loop. |
| Monitoring | DESIGNED | Offline metrics рассчитаны; production monitoring описан в MLSD. | Нужно реализовать сбор latency, drift, review rate, errors и RAG quality. |

## Обозначения

- **IMPLEMENTED** — компонент реально реализован и использовался в эксперименте.
- **IMPLEMENTED_AS_POLICY** — логика реализована на уровне experimental routing / evaluation policy.
- **PARTIAL** — ключевая идея реализована, но не в полном production-виде.
- **DESIGNED** — компонент описан в MLSD, но production infrastructure ещё не развёрнута.
- **PROPOSED** — компонент является рекомендуемым расширением архитектуры и не должен описываться как уже реализованный.

## Ключевой вывод

Prototype закрывает основную ML-часть задачи: intent classification, embeddings, semantic retrieval, Document RAG, LLM evaluation и selective routing.

Reranker, полноценная vector database и production monitoring относятся к production design. Tool Calling уже реализован через внутренний tool registry. Structured Output уже реализован через Pydantic. End-to-end LangGraph workflow уже реализован и проверен smoke-test.

В README и MLSD эти компоненты необходимо явно отделять от того, что было реально реализовано и измерено.

## LangGraph smoke-test

Исполняемый граф был проверен на четырёх сценариях:

- high-confidence intent → automatic resolution;
- low-confidence intent → human review;
- Document RAG → direct successful answer;
- Document RAG → additional search → successful answer.

Все smoke-tests завершились успешно.

Результаты: `results/langgraph_smoke_test.json`


## Structured Output implementation

Structured Output реализован через Pydantic v2.

Добавлены схемы:

- `IntentResult`;
- `RAGSource`;
- `AssistantResponse`.

Smoke-test проверяет:

- валидный intent-response;
- валидный Document RAG response;
- сохранение source metadata;
- human-review response;
- отклонение некорректного output;
- JSON serialization.

Все проверки завершились успешно.

Результаты: `results/structured_output_smoke_test.json`


## Tool Calling implementation

Tool Calling реализован через внутренний registry.

Добавлены tools:

- `get_account_status`;
- `get_product_info`.

Executor выполняет:

- проверку имени tool;
- Pydantic-валидацию аргументов;
- вызов зарегистрированной функции;
- возврат JSON-совместимого результата.

Smoke-test проверяет:

- успешный account tool call;
- успешный product tool call;
- отклонение неизвестного tool;
- отклонение некорректных аргументов;
- выдачу metadata registry;
- JSON serialization.

Все проверки завершились успешно.

Результаты: `results/tool_calling_smoke_test.json`
