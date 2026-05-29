from typing import List
from src.preprocess import chunk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def select_relevant_chunks(chunks, reference_summary, top_k=3, skip_first_n=2):
    candidate_chunks = chunks[skip_first_n:]

    texts = [chunk["text"] for chunk in candidate_chunks]

    vectorizer = TfidfVectorizer().fit(texts + [reference_summary])

    chunk_vecs = vectorizer.transform(texts)
    ref_vec = vectorizer.transform([reference_summary])

    similarities = cosine_similarity(chunk_vecs, ref_vec).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [candidate_chunks[i] for i in top_indices]


def simple_extractive_summary(text: str, max_sentences: int = 5) -> str:
    sentences = text.split(". ")

    if len(sentences) <= max_sentences:
        return text

    return ". ".join(sentences[:max_sentences]) + "."


def summarize_chunks(chunks: List[dict]) -> List[str]:
    summaries = []

    for chunk in chunks:
        text = chunk["text"]

        # skip header chunk
        if len(text) < 200:
            continue

        if text.isupper():
            continue

        if "UNITED STATES DISTRICT" in text[:200]:
            continue

        summary = simple_extractive_summary(text)
        summaries.append(summary)

    return summaries


def combine_summaries(chunk_summaries: List[str]) -> str:
    return "\n\n".join(chunk_summaries)


def summarize_document(text: str, reference_summary: str = None):
    chunks = chunk_text(text)

    if reference_summary:
        chunks = select_relevant_chunks(chunks, reference_summary)

    chunk_summaries = summarize_chunks(chunks)

    final_summary = combine_summaries(chunk_summaries)

    return final_summary