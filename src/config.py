"""Final frozen configuration of the project."""

SEED = 42

# Intent classification
DISTILBERT_MODEL = "distilbert-base-uncased"
NUM_LABELS = 77
NUM_EPOCHS = 6
MAX_LENGTH = 64

# Selected on validation
DISTILBERT_CONFIDENCE_THRESHOLD = 0.65

# Semantic retrieval
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Validation-optimal LangGraph gate.
# The value 1.0 means the low-confidence consensus branch
# did not provide safe additional coverage under the
# >=95% automatic-accuracy constraint.
RETRIEVAL_SIMILARITY_THRESHOLD = 1.0

# Document RAG
RAG_TOP_K = 3
RAG_GENERATOR_MODEL = "openai/gpt-oss-120b"

# Quality target
TARGET_AUTOMATIC_ACCURACY = 0.95