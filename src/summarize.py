from typing import List
from src.preprocess import chunk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def select_relevant_chunks(chunks, reference_summary, top_k=3, skip_first_n=2):
    candidate_chunks = chunks[skip_first_n:]

    texts = [chunk["text"] for chunk in candidate_chunks]

    vectorizer = TfidfVectorizer().fit(texts + [reference_summary])

    chunk_vecs = vectorizer.transform(texts)
    ref_vec = vectorizer.transform([reference_summary])

    similarities = cosine_similarity(chunk_vecs, ref_vec).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [candidate_chunks[i] for i in top_indices]



def simple_extractive_summary(
    text: str,
    max_sentences: int = 5
) -> str:

    # better sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        # skip if too short
        if len(sentence) < 40:
            continue

        # skip OCR noise
        weird_ratio = len(
            re.findall(r"[^a-zA-Z0-9\s,.]", sentence)
        ) / max(len(sentence), 1)

        if weird_ratio > 0.15:
            continue

        # skip headers / boilerplate
        if sentence.isupper():
            continue

        if "UNITED STATES DISTRICT" in sentence.upper():
            continue

        clean_sentences.append(sentence)

    if not clean_sentences:
        return text[:300]

    # tiny summary: choose one best sentence
    if max_sentences == 1:

        candidates = clean_sentences[:8]

        summary_keywords = [
            "lawsuit",
            "violation",
            "discrimination",
            "plaintiff",
            "defendant",
            "court",
            "filed",
            "alleged",
            "employment",
            "pregnancy",
            "civil rights",
            "injunction",
            "claim",
            "settlement",
            "complaint",
        ]

        scored_candidates = []

        for sentence in candidates:
            sentence_lower = sentence.lower()

            # filter incomplete OCR fragments
            words = sentence.split()

            if len(words) < 8:
                continue

            # filter incomplete OCR fragments like "Griffin due to..."
            if re.match(r"^[A-Z][a-z]+\s+due\s+to", sentence):
                continue

            if re.match(r"^[A-Z][a-z]+,\s", sentence):
                continue
            
            # likely truncated sentence
            if sentence[0].islower():
                continue

            # keyword score
            keyword_score = sum(
                keyword in sentence_lower
                for keyword in summary_keywords
            )

            # prefer medium-length readable sentence
            length_score = min(len(sentence) / 120, 1)

            total_score = keyword_score + length_score

            scored_candidates.append(
                (total_score, sentence)
            )

        if not scored_candidates:
            return clean_sentences[0]

        best_sentence = sorted(
            scored_candidates,
            key=lambda x: x[0],
            reverse=True
        )[0][1]

        best_sentence = best_sentence.strip()
        best_sentence = re.sub(r"\.{2,}", ".", best_sentence)

        return best_sentence


def summarize_chunks(
    chunks: List[dict],
    summary_type: str = "long"
) -> List[str]:

    summaries = []

    for chunk in chunks:
        text = chunk["text"]

        if len(text) < 200:
            continue

        if text.isupper():
            continue

        if "UNITED STATES DISTRICT" in text[:200]:
            continue

        summary = generate_summary(
            text,
            summary_type=summary_type
        )

        summaries.append(summary)

    return summaries


def combine_summaries(chunk_summaries: List[str]) -> str:
    return "\n\n".join(chunk_summaries)


def summarize_document(
    text: str,
    reference_summary: str = None,
    summary_type: str = "long",
    top_k: int = 3,
):
    chunks = chunk_text(text)

    if reference_summary:
        chunks = select_relevant_chunks(chunks, reference_summary, top_k=top_k)

    chunk_summaries = summarize_chunks(
        chunks,
        summary_type=summary_type
    )

    if summary_type == "tiny":
        return chunk_summaries[0] if chunk_summaries else ""

    final_summary = combine_summaries(chunk_summaries)
    return final_summary

def summarize_selected_chunks(chunks):
    combined_text = "\n\n".join(chunk["text"] for chunk in chunks)

    sentences = combined_text.split(". ")

    return ". ".join(sentences[:6]) + "."

def generate_summary(
    text: str,
    summary_type: str = "long",
):
    """
    summary_type:
    - long
    - short
    - tiny
    """

    if summary_type == "long":
        max_sentences = 6

    elif summary_type == "short":
        max_sentences = 3

    elif summary_type == "tiny":
        max_sentences = 1

    else:
        raise ValueError("summary_type must be long, short, or tiny")

    return simple_extractive_summary(
        text,
        max_sentences=max_sentences
    )