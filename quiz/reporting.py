from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _read_attempts(root: str | Path, quiz_id: str) -> list[dict[str, Any]]:
    base = Path(root) / "attempts" / quiz_id
    if not base.exists():
        return []
    attempts = []
    for path in sorted(base.glob("*.json")):
        try:
            attempts.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return attempts


def _pct(n: float, d: float) -> float:
    return round((n / d) * 100, 1) if d else 0.0


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else 0.0


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 1) if values else 0.0


def build_reports(root: str | Path, quiz_id: str) -> dict[str, list[dict[str, Any]]]:
    attempts = _read_attempts(root, quiz_id)
    student_rows = []
    question_rows = {}
    topic_rows = {}
    difficulty_rows = {}
    error_rows = []

    for attempt in attempts:
        questions = attempt.get("Questions", [])
        correct = sum(bool(q.get("Correct")) for q in questions)
        answered = sum(bool(q.get("Answered")) for q in questions)
        unanswered = len(questions) - answered
        total_time = sum(q.get("Time_Spent_Seconds") or 0 for q in questions)
        student_rows.append({
            "SAP_ID": attempt.get("SAP_ID", ""),
            "Quiz_ID": attempt.get("Quiz_ID", quiz_id),
            "Score": attempt.get("Score") if attempt.get("Score") is not None else correct,
            "Percentage": _pct(correct, len(questions)),
            "Correct": correct,
            "Incorrect": answered - correct,
            "Unanswered": unanswered,
            "Total_Time_Seconds": int(total_time),
            "Submitted_Timestamp": attempt.get("Submitted_Timestamp", ""),
            "Auto_Submitted": bool(attempt.get("Auto_Submitted")),
        })

        for q in questions:
            qid = q.get("Question_ID", "")
            topic = q.get("Topic", "")
            difficulty = q.get("Difficulty", "")
            answered_flag = bool(q.get("Answered"))
            correct_flag = bool(q.get("Correct"))
            time = q.get("Time_Spent_Seconds")
            changed = bool(q.get("Answer_Changed"))

            if qid not in question_rows:
                question_rows[qid] = {
                    "Question_ID": qid,
                    "Question_Family_ID": q.get("Question_Family_ID", ""),
                    "Topic": topic,
                    "Difficulty": difficulty,
                    "Students": 0,
                    "Attempted": 0,
                    "Correct": 0,
                    "Times": [],
                    "Unanswered": 0,
                    "Answer_Changes": 0,
                }
            r = question_rows[qid]
            r["Students"] += 1
            r["Attempted"] += int(answered_flag)
            r["Correct"] += int(correct_flag)
            r["Unanswered"] += int(not answered_flag)
            r["Answer_Changes"] += int(changed)
            if time is not None:
                r["Times"].append(float(time))

            for key, target in ((topic, topic_rows), (difficulty, difficulty_rows)):
                if key not in target:
                    target[key] = {
                        "Category": key,
                        "Students_Questions": 0,
                        "Attempted": 0,
                        "Correct": 0,
                        "Times": [],
                        "Unanswered": 0,
                        "Answer_Changes": 0,
                    }
                tr = target[key]
                tr["Students_Questions"] += 1
                tr["Attempted"] += int(answered_flag)
                tr["Correct"] += int(correct_flag)
                tr["Unanswered"] += int(not answered_flag)
                tr["Answer_Changes"] += int(changed)
                if time is not None:
                    tr["Times"].append(float(time))

            if (not answered_flag) or (answered_flag and not correct_flag):
                response = q.get("Final_Response") or ""
                if isinstance(response, list):
                    response = ", ".join(response)
                error_rows.append({
                    "SAP_ID": attempt.get("SAP_ID", ""),
                    "Question_ID": qid,
                    "Topic": topic,
                    "Difficulty": difficulty,
                    "Answered": answered_flag,
                    "Correct": correct_flag,
                    "Final_Response": response,
                    "Time_Spent_Seconds": q.get("Time_Spent_Seconds") or 0,
                    "Answer_Changed": changed,
                })

    question_report = []
    for r in question_rows.values():
        question_report.append({
            "Question_ID": r["Question_ID"],
            "Question_Family_ID": r["Question_Family_ID"],
            "Topic": r["Topic"],
            "Difficulty": r["Difficulty"],
            "Students": r["Students"],
            "Attempted": r["Attempted"],
            "Correct": r["Correct"],
            "% Correct": _pct(r["Correct"], r["Attempted"]),
            "% Unanswered": _pct(r["Unanswered"], r["Students"]),
            "Median_Time_Seconds": _median(r["Times"]),
            "Mean_Time_Seconds": _mean(r["Times"]),
            "Answer_Change_Rate": _pct(r["Answer_Changes"], r["Attempted"]),
        })

    def aggregate_report(source):
        output = []
        for r in source.values():
            output.append({
                "Category": r["Category"],
                "Attempted": r["Attempted"],
                "Correct": r["Correct"],
                "% Correct": _pct(r["Correct"], r["Attempted"]),
                "% Unanswered": _pct(r["Unanswered"], r["Students_Questions"]),
                "Median_Time_Seconds": _median(r["Times"]),
                "Mean_Time_Seconds": _mean(r["Times"]),
                "Answer_Change_Rate": _pct(r["Answer_Changes"], r["Attempted"]),
            })
        return output

    return {
        "student": sorted(student_rows, key=lambda x: x["SAP_ID"]),
        "question": sorted(question_report, key=lambda x: x["Question_ID"]),
        "topic": aggregate_report(topic_rows),
        "difficulty": aggregate_report(difficulty_rows),
        "errors": sorted(error_rows, key=lambda x: (x["SAP_ID"], x["Question_ID"])),
    }


def discuss_in_class(
    question_report: list[dict[str, Any]],
    *,
    low_accuracy: float = 60.0,
    high_time_seconds: float = 75.0,
    high_change_rate: float = 35.0,
    high_accuracy: float = 85.0,
    low_time_seconds: float = 35.0,
    very_low_time_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    flags = []
    for q in question_report:
        accuracy = float(q["% Correct"])
        median_time = float(q["Median_Time_Seconds"])
        changes = float(q["Answer_Change_Rate"])
        label = None
        interpretation = None
        if accuracy < low_accuracy and median_time >= high_time_seconds:
            label = "Low accuracy + high time"
            interpretation = "Students struggled despite spending substantial time."
        elif accuracy < low_accuracy and changes >= high_change_rate:
            label = "Low accuracy + high answer changes"
            interpretation = "Students showed uncertainty or competing interpretations."
        elif accuracy >= high_accuracy and median_time <= low_time_seconds:
            label = "High accuracy + low time"
            interpretation = "Concept appears well mastered."
        elif accuracy < low_accuracy and median_time <= very_low_time_seconds:
            label = "Low accuracy + very low time"
            interpretation = "Possible superficial reading, guessing, or misconception."
        if label:
            flags.append({
                "Question_ID": q["Question_ID"],
                "Topic": q["Topic"],
                "Difficulty": q["Difficulty"],
                "Flag": label,
                "% Correct": accuracy,
                "Median_Time_Seconds": median_time,
                "Answer_Change_Rate": changes,
                "Interpretation": interpretation,
            })
    return flags


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
