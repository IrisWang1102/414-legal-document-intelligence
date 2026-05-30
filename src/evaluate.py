from rouge_score import rouge_scorer


def evaluate_summary(pred_summary: str, reference_summary: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference_summary, pred_summary)

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }