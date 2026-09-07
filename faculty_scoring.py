"""Transparent, faculty-reviewable provisional scoring for the ML assignment.

The engine is intentionally conservative: it deducts marks only for evidence-backed
issues.  Faculty-review items remain review flags and do not automatically reduce
marks.  Every issue gets Observation / Why it matters / Prescription feedback.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import openpyxl
except Exception:  # pragma: no cover
    openpyxl = None

RUBRIC_MAX = {
    "Business understanding & ML formulation": 3.0,
    "Data understanding & EDA": 5.0,
    "Data preparation": 4.0,
    "Model selection & implementation": 6.0,
    "Model evaluation & comparison": 5.0,
    "Business interpretation": 4.0,
    "Critical reflection & reproducibility": 3.0,
}

SHEET_RUBRIC = {
    "EDA_Results": "Data understanding & EDA",
    "Multicollinearity": "Data preparation",
    "Dimensionality_Reduction": "Data preparation",
    "Clustering": "Model selection & implementation",
    "Model_Evaluation": "Model evaluation & comparison",
    "Business_Interpretation": "Business interpretation",
    "Submission_Check": "Critical reflection & reproducibility",
}


@dataclass
class Issue:
    category: str
    rubric: str
    deduction: float
    severity: str
    observation: str
    why_it_matters: str
    prescription: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    s = str(value).strip().replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None


def _sheet_rows(ws) -> list[list[Any]]:
    return [[c.value for c in row] for row in ws.iter_rows() if any(c.value is not None for c in row)]


def _flatten(rows: list[list[Any]]) -> str:
    return "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)


def _find_text(rows: list[list[Any]], needle: str) -> list[str]:
    needle = needle.lower()
    hits = []
    for row in rows:
        s = " | ".join("" if v is None else str(v) for v in row)
        if needle in s.lower():
            hits.append(s)
    return hits


def _extract_zip(zip_path: str | Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        out = {}
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name
            # Ignore Mac metadata and checkpoint copies for content analysis.
            if "__MACOSX" in name or ".ipynb_checkpoints" in name:
                continue
            out[name] = zf.read(name)
        return out


def _load_materials(zip_path: str | Path):
    files = _extract_zip(zip_path)
    workbook = None
    workbook_bytes = next((b for n, b in files.items() if n.lower().endswith(".xlsx")), None)
    if workbook_bytes is not None and openpyxl is not None:
        workbook = openpyxl.load_workbook(io.BytesIO(workbook_bytes), data_only=False)

    notebook_text = ""
    notebook_json = None
    nb_bytes = next((b for n, b in files.items() if n.lower().endswith(".ipynb")), None)
    if nb_bytes is not None:
        try:
            notebook_json = json.loads(nb_bytes.decode("utf-8"))
            pieces = []
            for cell in notebook_json.get("cells", []):
                src = "".join(cell.get("source", []))
                pieces.append(src)
            notebook_text = "\n\n".join(pieces)
        except Exception:
            notebook_text = nb_bytes.decode("utf-8", errors="ignore")

    pdf_text = ""
    pdf_bytes = next((b for n, b in files.items() if n.lower().endswith(".pdf")), None)
    if pdf_bytes:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pdf_text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            pdf_text = ""

    return files, workbook, notebook_text, pdf_text


def _issues_from_evidence(evaluation_rows: list[dict[str, str]]) -> list[Issue]:
    issues: list[Issue] = []
    for row in evaluation_rows:
        if str(row.get("verified", "")).strip() != "False":
            continue
        check = _norm(row.get("check"))
        reported = row.get("reported", "")
        actual = row.get("actual", "")
        rep_n = _num(reported)
        act_n = _num(actual)

        # These are strong, low-ambiguity evidence failures.
        if "number of observations" in check and rep_n is not None and act_n is not None and rep_n != act_n:
            issues.append(Issue(
                "ERROR", "Data understanding & EDA", 0.5, "HIGH",
                f"Reported observations ({reported}) do not match the dataset ({actual}).",
                "Incorrect dataset-size reporting weakens the reliability of the EDA and any downstream calculations.",
                "Re-run the dataset structure checks directly on the assigned dataset and reconcile the reported value."
            ))
        elif "number of variables" in check and rep_n is not None and act_n is not None and rep_n != act_n:
            issues.append(Issue(
                "ERROR", "Data understanding & EDA", 0.5, "MEDIUM",
                f"Reported variable count ({reported}) does not match the dataset ({actual}).",
                "An incorrect variable count can indicate that predictors or the target were included/excluded inconsistently.",
                "Verify the target/predictor split and report the dataset dimensions exactly as used in the analysis."
            ))
        elif "missing values" in check and rep_n is not None and act_n is not None and rep_n != act_n:
            issues.append(Issue(
                "ERROR", "Data understanding & EDA", 0.5, "HIGH",
                f"Reported missing values ({reported}) do not match the verified dataset value ({actual}).",
                "Missing-data handling affects preprocessing and can change model results.",
                "Recalculate missing values from the assigned dataset and document the treatment explicitly."
            ))
        elif "duplicate observations" in check and rep_n is not None and act_n is not None and rep_n != act_n:
            issues.append(Issue(
                "ERROR", "Data understanding & EDA", 0.5, "HIGH",
                f"Reported duplicates ({reported}) do not match the verified dataset value ({actual}).",
                "Duplicate records can distort descriptive statistics and model training.",
                "Recalculate duplicates and explain whether duplicates were removed or retained and why."
            ))
        elif "highest predictor correlation" in check and rep_n is not None and act_n is not None:
            if abs(rep_n - act_n) > 0.01:
                issues.append(Issue(
                    "ERROR", "Data understanding & EDA", 0.5, "MEDIUM",
                    f"Reported highest predictor correlation ({reported}) differs materially from the verified value ({actual}).",
                    "Correlation evidence is used to diagnose predictor redundancy and should be numerically reproducible.",
                    "Recompute the predictor correlation matrix and verify the reported maximum pair and value."
                ))
        elif "highest predictor vif" in check:
            if rep_n is None and act_n is not None:
                issues.append(Issue(
                    "OMISSION", "Data preparation", 0.5, "HIGH",
                    f"The submission did not report a usable VIF value; the evaluator verified {actual}.",
                    "VIF is an important diagnostic for multicollinearity in this assignment.",
                    "Compute and report the highest predictor VIF, identify the variable, and explain the decision taken."
                ))
            elif rep_n is not None and act_n is not None and abs(rep_n - act_n) > 0.10:
                issues.append(Issue(
                    "ERROR", "Data preparation", 0.5, "HIGH",
                    f"Reported highest VIF ({reported}) differs materially from the verified value ({actual}).",
                    "An incorrect VIF can lead to an incorrect conclusion about multicollinearity and remediation.",
                    "Recompute VIF using the same predictor set used for modelling and reconcile the reported value."
                ))
        elif "variables affected by multicollinearity" in check:
            # Count/threshold wording often differs legitimately, so keep this as a
            # modest deduction plus faculty-review note rather than a hard penalty.
            rep_n_first = _num(reported)
            act_n_first = _num(actual)
            if rep_n_first is None:
                continue
            if act_n_first is not None and rep_n_first != act_n_first:
                issues.append(Issue(
                    "ERROR", "Data preparation", 0.5, "MEDIUM",
                    f"The reported number of variables affected by multicollinearity ({reported}) does not match the verified count ({actual}).",
                    "The severity and scope of multicollinearity influence whether a remedy is needed.",
                    "Report the count using the evaluator's threshold definition and reconcile any distinction between moderate and severe VIF cut-offs."
                ))
        elif "original number of dimensions" in check and rep_n is not None and act_n is not None and rep_n != act_n:
            issues.append(Issue(
                "ERROR", "Data understanding & EDA", 0.5, "MEDIUM",
                f"Reported predictor dimensions ({reported}) do not match the verified dimensionality ({actual}).",
                "Dimensionality is the baseline for deciding whether reduction is warranted.",
                "Reconcile the predictor count after separating the target and any categorical variable."
            ))
        elif "roc-auc" in check or "r-squared" in check or "rmse" in check or "model selection / evaluation" in check:
            issues.append(Issue(
                "ERROR", "Model evaluation & comparison", 0.5, "HIGH",
                f"The reported model-evaluation evidence ({reported}) was not independently verified against {actual}.",
                "Model-selection claims should be supported by reproducible validation and evaluation results.",
                "Re-run the stated validation procedure, verify the primary metric, and reconcile the reported result with the analysis output."
            ))
        elif "classification evaluation metric" in check:
            issues.append(Issue(
                "OMISSION", "Model evaluation & comparison", 0.5, "HIGH",
                f"The primary classification evaluation metric was not sufficiently evidenced: {reported}.",
                "A suitable metric is required to connect the model to the business objective, especially with class imbalance.",
                "State the primary metric explicitly and explain why it is appropriate for the decision problem."
            ))
    return issues


def _section_quality_issues(workbook, notebook_text, pdf_text) -> tuple[list[Issue], list[Issue]]:
    issues: list[Issue] = []
    review_flags: list[Issue] = []
    corpus = (notebook_text + "\n" + pdf_text).lower()

    # Business understanding: conservative missing-evidence flag only.
    business_terms = ["business problem", "business objective", "analytical objective", "decision", "target"]
    if not any(t in corpus for t in business_terms):
        issues.append(Issue(
            "OMISSION", "Business understanding & ML formulation", 0.5, "MEDIUM",
            "No clear business-problem/analytical-objective language was detected in the submitted analysis text.",
            "The assignment assesses whether the ML formulation follows from the business problem rather than from an algorithm choice.",
            "State the business decision, analytical objective, target/outcome and why the chosen ML framing supports that decision."
        ))

    # Model development: require evidence of more than a single arbitrary model.
    if workbook and "Model_Evaluation" in workbook.sheetnames:
        rows = _sheet_rows(workbook["Model_Evaluation"])
        text = _flatten(rows).lower()
        candidate_markers = ["logistic regression", "linear regression", "random forest", "decision tree", "gradient boosting", "k-means", "hierarchical"]
        model_count = sum(1 for m in candidate_markers if m in text)
        selected = any("selected approach" in r.lower() for r in _find_text(rows, "selected approach"))
        rationale = any("why this approach was selected" in r.lower() for r in _find_text(rows, "why this approach was selected"))
        if model_count < 2 and not ("baseline" in text and selected):
            issues.append(Issue(
                "SUBOPTIMAL DECISION", "Model selection & implementation", 0.5, "MEDIUM",
                "The submission provides limited evidence of meaningful model comparison or baseline reasoning.",
                "Model selection should be justified against alternatives, not only by presenting one algorithm.",
                "Compare an appropriate baseline and alternatives, then justify the final model using validation evidence and the business objective."
            ))
        elif not rationale:
            issues.append(Issue(
                "OMISSION", "Model selection & implementation", 0.5, "MEDIUM",
                "A final model is identified, but an explicit selection rationale was not detected in the template.",
                "The rubric rewards analytical reasoning behind model choice, not algorithm names alone.",
                "Add a concise rationale linking model choice to validation performance, interpretability and the business problem."
            ))

        eval_text = text
        if not any(t in eval_text for t in ["cross-validation", "cross validation", "train/test", "holdout", "validation"]):
            issues.append(Issue(
                "OMISSION", "Model evaluation & comparison", 0.5, "HIGH",
                "No clear validation strategy was detected in the reported model-evaluation evidence.",
                "Without appropriate validation, reported performance may be optimistic or unstable.",
                "Use a defensible train/test or cross-validation strategy and state exactly how it was used."
            ))

    # Business interpretation: require findings and recommendations, not metrics only.
    if workbook and "Business_Interpretation" in workbook.sheetnames:
        rows = _sheet_rows(workbook["Business_Interpretation"])
        text = _flatten(rows).lower()
        finding_rows = [r for r in rows if r and isinstance(r[0], str) and r[0].strip().lower() not in {"business interpretation", "key findings", "recommendations", "recommendation", "finding"}]
        if len(finding_rows) < 2 or "business meaning" not in text or "implication" not in text:
            issues.append(Issue(
                "OMISSION", "Business interpretation", 0.5, "MEDIUM",
                "Business interpretation evidence appears limited or lacks a clear chain from analytical finding to business implication.",
                "The assignment requires translation of model results into business decisions and limitations.",
                "For the major findings, state the evidence, business meaning, implication/recommendation and relevant limitation."
            ))
        if "recommendation" not in text:
            issues.append(Issue(
                "OMISSION", "Business interpretation", 0.5, "MEDIUM",
                "No explicit recommendations were detected in the business interpretation section.",
                "Metrics alone do not demonstrate how the analysis supports a business decision.",
                "Translate the strongest analytical findings into specific, evidence-based recommendations."
            ))

    # Critical reflection / reproducibility: conservative keyword test.
    reflection_terms = ["limitation", "reflection", "alternative", "reproduc", "random_state", "seed", "ai", "chatgpt", "copilot", "gemini"]
    hits = sum(t in corpus for t in reflection_terms)
    if hits < 3:
        issues.append(Issue(
            "OMISSION", "Critical reflection & reproducibility", 0.5, "MEDIUM",
            "Limited evidence was detected for reflection, reproducibility and/or validation of AI-assisted suggestions.",
            "The rubric explicitly assesses reflection on analytical decisions and reproducibility of the submitted work.",
            "Document the main analytical challenge, a key decision and alternative considered, how AI-assisted suggestions were validated, and enough detail to reproduce the analysis."
        ))

    # Generic unresolved flag heuristic. This is kept as a faculty-review deduction
    # because a flag can be legitimate when later analysis explicitly addresses it.
    flag_patterns = re.findall(r"[\"']([A-Za-z][A-Za-z0-9_]*_Flag)[\"'](?:\s*\])?\s*=", notebook_text)
    for flag in sorted(set(flag_patterns)):
        # Look for an actual row-level treatment of the flagged observations.
        lower_nb = notebook_text.lower()
        flag_lower = flag.lower()
        handled_patterns = [
            f'~df_clean["{flag_lower}"]',
            f'~df_clean[\'{flag_lower}\']',
            f'df_clean.loc[df_clean["{flag_lower}"] == false]',
            f'df_clean.loc[df_clean["{flag_lower}"] == true]',
            f'~{flag_lower}',
        ]
        explicitly_handled = any(p in lower_nb for p in handled_patterns)

        # A flag that is calculated but never used for filtering/cleaning is a
        # useful faculty-review prompt even when the rest of the submission is strong.
        if not explicitly_handled:
            review_flags.append(Issue(
                "OMISSION", "Data preparation", 0.0, "MEDIUM",
                f"A data-quality flag ({flag}) is calculated; faculty should verify that the flagged observations were appropriately resolved, retained with justification, or tested for sensitivity.",
                "Identifying an anomaly is only part of data preparation; its analytical impact should be resolved or justified.",
                f"Explain why {flag} observations were retained/removed and, where relevant, test whether the modelling conclusions change."
            ))
            break
    return issues, review_flags


def _build_strengths(workbook, evaluation_rows, notebook_text):
    strengths = []
    checks = [r for r in evaluation_rows if str(r.get("verified", "")) == "True"]
    if len(checks) >= 5:
        strengths.append("Core reported dataset and analytical evidence is internally consistent with the independent checks.")
    if workbook and "Model_Evaluation" in workbook.sheetnames:
        text = _flatten(_sheet_rows(workbook["Model_Evaluation"])).lower()
        if "cross-validation" in text or "cross validation" in text:
            strengths.append("Model comparison uses explicit validation evidence rather than relying on a single model.")
    if workbook and "Business_Interpretation" in workbook.sheetnames:
        text = _flatten(_sheet_rows(workbook["Business_Interpretation"])).lower()
        if "business meaning" in text and "recommendation" in text:
            strengths.append("The submission translates analytical findings into business implications and recommendations.")
    if any(t in notebook_text.lower() for t in ["random_state", "random state", "set.seed"]):
        strengths.append("Reproducibility evidence is present through an explicit random seed/state.")
    return strengths[:4]


def score_submission(zip_path: str | Path, evaluation_rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    files, workbook, notebook_text, pdf_text = _load_materials(zip_path)
    evaluation_rows = evaluation_rows or []
    issues = _issues_from_evidence(evaluation_rows)
    section_issues, review_flags = _section_quality_issues(workbook, notebook_text, pdf_text)
    issues += section_issues

    # De-duplicate automated deductions by rubric + observation.
    seen = set()
    deduped = []
    for issue in issues:
        key = (issue.rubric, issue.observation)
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    issues = deduped

    # De-duplicate faculty-review flags.
    seen_flags = set()
    unique_flags = []
    for flag in review_flags:
        key = (flag.rubric, flag.observation)
        if key not in seen_flags:
            seen_flags.add(key)
            unique_flags.append(flag)
    review_flags = unique_flags

    scores = dict(RUBRIC_MAX)
    for issue in issues:
        scores[issue.rubric] = max(0.0, scores[issue.rubric] - issue.deduction)

    total = round(sum(scores.values()), 1)
    # Confidence is lower when there are faculty-review items or more than one issue.
    faculty_review_count = sum(str(r.get("verified", "")) not in ("True", "False") for r in evaluation_rows)
    hard_issues = len(issues)
    total_review_flags = len(review_flags)
    if hard_issues == 0 and total_review_flags == 0 and faculty_review_count <= 2:
        confidence = "High"
        priority = "Low"
    elif hard_issues <= 2 and total_review_flags <= 2 and faculty_review_count <= 3:
        confidence = "Moderate"
        priority = "Low–Moderate"
    else:
        confidence = "Low"
        priority = "High"

    comments = []
    for issue in issues:
        comments.append(
            f"[{issue.category}] {issue.observation} Why it matters: {issue.why_it_matters} Prescription: {issue.prescription}"
        )

    # Student-facing draft: strengths + actionable improvements.
    strengths = _build_strengths(workbook, evaluation_rows, notebook_text)
    feedback_parts = []
    if strengths:
        feedback_parts.append("WHAT YOU DID WELL\n" + "\n".join(f"• {s}" for s in strengths))
    if issues:
        feedback_parts.append("WHAT COULD BE IMPROVED\n" + "\n".join(
            f"• Observation: {i.observation}\n  Why it matters: {i.why_it_matters}\n  How to improve: {i.prescription}" for i in issues
        ))
    else:
        feedback_parts.append("WHAT COULD BE IMPROVED\n• No substantive automated improvement flag was identified. Faculty may still refine the feedback based on the full submission.")

    at_risk_points = round(sum(f.deduction for f in review_flags), 1)

    return {
        "rubric_scores": scores,
        "provisional_score": total,
        "confidence": confidence,
        "review_priority": priority,
        "hard_issue_count": len(issues),
        "faculty_review_count": faculty_review_count,
        "review_flags": [i.as_dict() for i in review_flags],
        "at_risk_points": at_risk_points,
        "issues": [i.as_dict() for i in issues],
        "strengths": strengths,
        "faculty_comments": "\n\n".join(comments),
        "student_feedback": "\n\n".join(feedback_parts),
        "file_count": len(files),
    }
