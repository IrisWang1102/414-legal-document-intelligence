import re


CLAUSE_KEYWORDS = {
    "payment": [
        "payment",
        "fee",
        "invoice",
        "penalty",
        "late payment"
    ],
    "termination": [
        "termination",
        "terminate",
        "written notice",
        "effective"
    ],
    "liability": [
        "liability",
        "damages",
        "liable",
        "legal advice"
    ],
    "confidentiality": [
        "confidential",
        "confidentiality",
        "disclosure",
        "third parties"
    ],
    "compliance": [
        "compliance",
        "regulation",
        "law",
        "governed"
    ]
}


def summarize_document(text: str) -> str:
    """
    Simple rule-based summary:
    return first few important sentences.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    summary_sentences = sentences[:5]

    return " ".join(summary_sentences)


def extract_clauses(text: str) -> dict:
    """
    Extract clause-like sentences
    using keyword matching.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    clauses = {}

    for clause_name, keywords in CLAUSE_KEYWORDS.items():

        matched_sentences = []

        for sentence in sentences:

            sentence_lower = sentence.lower()

            if any(k in sentence_lower for k in keywords):
                matched_sentences.append(sentence)

        if matched_sentences:
            clauses[clause_name] = " ".join(matched_sentences[:3])
        else:
            clauses[clause_name] = "Not found."

    return clauses