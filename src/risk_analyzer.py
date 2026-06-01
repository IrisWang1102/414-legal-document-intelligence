def analyze_risks(text: str, clauses: dict) -> list[dict]:

    risk_table = []

    for clause_name, clause_text in clauses.items():

        clause_lower = clause_text.lower()

        risk_level = "Low"
        explanation = "Low legal concern."

        if any(
            keyword in clause_lower
            for keyword in [
                "penalty",
                "terminate immediately",
                "damages",
                "liable",
                "not exceed",
                "inaccuracy",
                "breach"
            ]
        ):
            risk_level = "High"
            explanation = (
                "Potential legal or financial exposure detected."
            )

        elif any(
            keyword in clause_lower
            for keyword in [
                "notice",
                "confidential",
                "compliance",
                "regulation"
            ]
        ):
            risk_level = "Medium"
            explanation = (
                "Moderate compliance or operational risk."
            )

        risk_table.append(
            {
                "Risk Area": clause_name.capitalize(),
                "Risk Level": risk_level,
                "Evidence": clause_text,
                "Explanation": explanation
            }
        )

    return risk_table