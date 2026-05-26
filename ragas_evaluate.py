import json
import os
import time
import numpy as np
import faiss
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from kashivani.generator import generate_answer
from kashivani.ingestor import load_pdfs
from kashivani.chunker import chunk_documents

from ragas import (
    EvaluationDataset,
    SingleTurnSample,
    evaluate
)

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    FactualCorrectness
)

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from openai import AsyncOpenAI


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

# Start with ONE model first
# Add more later after successful run

MODELS_TO_TEST = [
    "all-MiniLM-L6-v2"
]

EVAL_PATH = "data/eval_questions_clean.json"

TOP_K = 3

USE_GPU = True

REQUEST_DELAY = 5

RETRY_DELAY = 10

MAX_RETRIES = 5


# ============================================================
# BUILD TEMP FAISS INDEX
# ============================================================

def build_temp_index(model, chunks):

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_with_model(
    query,
    model,
    index,
    chunks,
    top_k=TOP_K
):

    query_vector = model.encode([query])

    query_vector = np.array(
        query_vector
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    return [chunks[idx] for idx in indices[0]]


# ============================================================
# LOAD EMBEDDING MODEL SAFELY
# ============================================================

def load_embedding_model(model_name):

    if USE_GPU and torch.cuda.is_available():

        print("Using GPU (CUDA)")

        model = SentenceTransformer(
            model_name,
            device="cuda"
        )

    else:

        print("Using CPU")

        model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    return model


# ============================================================
# RAGAS EVALUATOR LLM
# ============================================================

def make_evaluator_llm():

    async_client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    return llm_factory(
        "llama-3.1-8b-instant",
        client=async_client
    )


# ============================================================
# MODERN RAGAS EMBEDDINGS
# ============================================================

def make_evaluator_embeddings():

    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    return RagasHFEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# SAFE ANSWER GENERATION WITH RETRIES
# ============================================================

def safe_generate_answer(
    query,
    retrieved_chunks
):

    retries = 0

    while retries < MAX_RETRIES:

        try:

            answer = generate_answer(
                query,
                retrieved_chunks
            )

            return answer

        except Exception as e:

            retries += 1

            print("\n" + "=" * 60)
            print("RATE LIMIT / API ERROR")
            print("=" * 60)

            print(e)

            print(
                f"\nRetry "
                f"{retries}/{MAX_RETRIES}"
            )

            print(
                f"Waiting "
                f"{RETRY_DELAY} seconds..."
            )

            time.sleep(RETRY_DELAY)

    return "Failed to generate answer."


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_model_with_ragas(
    model_name,
    questions,
    chunks
):

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print("Loading embedding model...")

    model = load_embedding_model(model_name)

    # --------------------------------------------------------
    # BUILD FAISS INDEX
    # --------------------------------------------------------

    print("Building index...")

    index = build_temp_index(
        model,
        chunks
    )

    # --------------------------------------------------------
    # GENERATE ANSWERS
    # --------------------------------------------------------

    samples = []

    print(
        f"Running {len(questions)} queries..."
    )

    for i, q in enumerate(questions):

        query = q["question"]

        reference = q["reference"]

        # --------------------------------------------
        # RETRIEVE CHUNKS
        # --------------------------------------------

        retrieved_chunks = retrieve_with_model(
            query,
            model,
            index,
            chunks
        )

        # --------------------------------------------
        # LIMIT CONTEXT SIZE
        # --------------------------------------------

        contexts = [
            c["text"][:1200]
            for c in retrieved_chunks
        ]

        # --------------------------------------------
        # GENERATE ANSWER SAFELY
        # --------------------------------------------

        answer = safe_generate_answer(
            query,
            retrieved_chunks
        )

        # --------------------------------------------
        # CREATE SAMPLE
        # --------------------------------------------

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=contexts,
            response=answer,
            reference=reference
        )

        samples.append(sample)

        print(
            f"  [{i+1}/{len(questions)}] "
            f"{query[:60]}..."
        )

        # --------------------------------------------
        # PREVENT RATE LIMITS
        # --------------------------------------------

        time.sleep(REQUEST_DELAY)

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    dataset = EvaluationDataset(
        samples=samples
    )

    # --------------------------------------------------------
    # INITIALISE RAGAS
    # --------------------------------------------------------

    print("\nInitialising RAGAS metrics...")

    evaluator_llm = make_evaluator_llm()

    evaluator_embeddings = (
        make_evaluator_embeddings()
    )

    metrics = [

        Faithfulness(
            llm=evaluator_llm
        ),

        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ),

        FactualCorrectness(
            llm=evaluator_llm
        )
    ]

    # --------------------------------------------------------
    # RUN RAGAS
    # --------------------------------------------------------

    print("Running RAGAS evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=metrics
    )

    # --------------------------------------------------------
    # EXTRACT SCORES
    # --------------------------------------------------------

    scores = {

        "model": model_name,

        "faithfulness": round(
            float(results["faithfulness"]),
            4
        ),

        "answer_relevancy": round(
            float(results["answer_relevancy"]),
            4
        ),

        "factual_correctness": round(
            float(results["factual_correctness"]),
            4
        )
    }

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"\nResults for {model_name}:")

    print(
        f"  Faithfulness: "
        f"{scores['faithfulness']}"
    )

    print(
        f"  Answer Relevancy: "
        f"{scores['answer_relevancy']}"
    )

    print(
        f"  Factual Correctness: "
        f"{scores['factual_correctness']}"
    )

    return scores


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    print(
        "Loading documents and chunks..."
    )

    docs = load_pdfs("data/raw")

    chunks = chunk_documents(docs)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    print(
        f"Loaded {len(questions)} "
        f"clean evaluation questions"
    )

    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    all_results = []

    for model_name in MODELS_TO_TEST:

        result = evaluate_model_with_ragas(
            model_name,
            questions,
            chunks
        )

        all_results.append(result)

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("FINAL RAGAS COMPARISON")
    print(f"{'=' * 60}")

    print(
        f"{'Model':<45} "
        f"{'Faith':>10} "
        f"{'Relev':>10} "
        f"{'Fact':>10}"
    )

    print("-" * 80)

    for r in all_results:

        print(
            f"{r['model']:<45} "
            f"{r['faithfulness']:>10} "
            f"{r['answer_relevancy']:>10} "
            f"{r['factual_correctness']:>10}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    with open(
        "data/ragas_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to "
        "data/ragas_results.json"
    )