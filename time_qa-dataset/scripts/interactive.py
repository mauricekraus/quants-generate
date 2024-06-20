"""
Used for interactively debugging the generated dataset.

Usage:
    Change the distribution (`question_type_distribution`) to whatever you want in `generate/prompts/run.py`,
    e.g. to `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

    Run the following commands iteratively to generate the dataset and interactively debug it:
    if [ -d generated-dataset-debugging ]; then rm -R generated-dataset-debugging/ ; fi && python -m generate prompts --num-samples 15 --num-qapairs 1 --print-first 0 --output-dir generated-dataset-debugging --answer-types multi && python scripts/interactive.py
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from rich.console import Console

if __name__ == "__main__":
    data_dir = Path("generated-dataset-debugging") / "data"

    class Sample(NamedTuple):
        sample_id: int
        question_id: int
        action_sequence: dict[str, Any]
        textual_description: str
        question_type: str
        question: str
        answer_type: str
        answer: str
        options: dict[str, str | bool] | None
        correct_option: str

    def get_for(num: int):
        path = data_dir / str(num)
        with open(path / "data.json") as json_file:
            data = json.load(json_file)

        return [
            Sample(
                sample_id=num,
                question_id=question_id,
                action_sequence=data["prompt_sequence"],
                textual_description=data["textual_description"],
                question_type=qa_pair["question_type"],
                question=qa_pair["question"],
                answer_type=qa_pair["answer_type"],
                answer=qa_pair["answer"],
                options=qa_pair["options"],  # can be None
                correct_option=qa_pair["correct_option"],
            )
            for question_id, qa_pair in enumerate(data["qa_pairs"])
        ]

    all = [sample for path in data_dir.iterdir() for sample in get_for(int(path.name))]

    df = pd.DataFrame(all).groupby("question_type")

    console = Console()

    for question_type in df.groups:
        console.rule(f"[bold red]{question_type}")

        for element in df.get_group(question_type).itertuples():
            console.print(f"[bold blue]Sample ID: {element.sample_id}, Question ID: {element.question_id}")
            console.print(f"Question: {element.question}")
            console.print(f"Action Sequence: {[elem['action'] for elem in element.action_sequence]}")
            console.print(f"Answer: {element.answer}")
            console.print(f"Options: {element.options}")
            console.print(f"Correct Option: {element.correct_option}")
            console.print()
