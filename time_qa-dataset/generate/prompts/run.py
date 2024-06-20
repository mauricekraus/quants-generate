import json
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import typer
from numpy.random import default_rng
from rich.console import Console
from rich.progress import Progress

from .base.sample import sample_data_instance
from .base.utility import InvalidPromptSequenceError, add_dicts, filter_dict_by_keys


class AnswerType(str, Enum):
    all = "all"
    open = "open"
    binary = "binary"
    multi = "multi"


def range_assert_callback(ctx: typer.Context, value: int):
    if ctx.resilient_parsing:
        return
    f_min = ctx.params["similarity_factor_min"]
    f_max = ctx.params["similarity_factor_max"]
    min_actions = ctx.params["minimum_actions"]

    if f_max < f_min:
        raise typer.BadParameter(
            "similarity_factor_max must be greater than or equal to similarity_factor_min"
        )
    if f_max > min_actions:
        raise typer.BadParameter("similarity_factor_max must be less than or equal to minimum_actions")

    return value


def output_callback(ctx: typer.Context, value: Path):
    if value.exists() and value.is_dir() and any(True for _ in value.iterdir()):
        raise typer.BadParameter(
            f"Directory {value} already exists and is not empty. If regeneration is intentional, empty it first."
        )

    (value / "data").mkdir(exist_ok=True, parents=True)
    return value


def generate(
    answer_types: AnswerType = typer.Option(AnswerType.all, help="Type of answers"),
    num_samples: int = typer.Option(100, min=1, help="Size of the generated dataset."),
    print_first: int = typer.Option(
        3, min=0, help="Number of data instances to print colorfully in the terminal."
    ),
    minimum_actions: int = typer.Option(4, min=1, help="Minimum number of actions in a data instance."),
    maximum_actions: int = typer.Option(4, min=1, help="Maximum number of actions in a data instance."),
    similarity_factor_min: int = typer.Option(
        1, min=0, help="Minimum similarity factor for question generation."
    ),
    similarity_factor_max: int = typer.Option(
        2, min=1, help="Maximum similarity factor for question generation."
    ),
    num_qapairs: int = typer.Option(5, min=1, help="Number of question-answer pairs in a data instance."),
    time_series_length: float = typer.Option(
        12.0, min=0.1, help="Total length of the time series in seconds."
    ),
    minimum_fraction_of_time_per_step: float = typer.Option(
        1.0,
        min=0.1,
        help="Minimum fraction of time per step, ensuring each step is at least as long as this fraction.",
    ),
    question_type_distribution: list[float] = typer.Option(
        default_factory=lambda: [6.0, 6.0, 3.0, 3.0, 5.0, 3.0, 5.0, 5.0, 5.0, 3.0],
        help="Distribution of question types in the dataset default: [6.0, 6.0, 3.0, 3.0, 5.0, 3.0, 5.0, 5.0, 5.0, 3.0].",
    ),
    seed: Optional[int] = typer.Option(
        1,
        min=0,
        help="Seed for random number generation to ensure reproducibility.",
        callback=range_assert_callback,
    ),  # callback need to go last
    output_dir: Path = typer.Option(
        Path("generated-dataset"),
        help="Directory where the generated dataset will be saved.",
        callback=output_callback,
    ),
    max_attempts: int = 1000,
):
    """
    Generate a custom dataset based on the provided parameters.
    """

    typer.echo(f"Generating dataset with {num_samples} samples...")

    rng = default_rng(seed)
    console = Console()

    # this section loops the whole generation process
    stats = {
        "InvalidPromptSequenceErrors": 0,
        "question_types": {},
        "answer_types": {},
        "correct_answers": {},
    }
    index = 0
    try:
        with Progress(expand=True, speed_estimate_period=2, console=console) as progress:
            sampling_task = progress.add_task("[yellow]Sampling...", total=num_samples)

            attempts = 0

            while index < num_samples:
                try:
                    instance = sample_data_instance(
                        index=index,
                        min_actions=minimum_actions,
                        max_actions=maximum_actions,
                        entire_time_length=time_series_length,
                        rng=rng,
                        num_qa_pairs_each=num_qapairs,
                        min_fraction_of_time_per_step=minimum_fraction_of_time_per_step,
                        similarity_factor_min=similarity_factor_min,
                        similarity_factor_max=similarity_factor_max,
                        answer_types=answer_types,
                        question_type_distribution=question_type_distribution,
                        distinct_actions=False,
                        distinct_templates=False,
                    )
                    if index < print_first:
                        instance.print(console)
                    instance.dataset_dir = output_dir / "data"
                    instance.to_disk()

                    # update stats dict with values of current iteration
                    stats["question_types"] = add_dicts(
                        stats["question_types"], instance.stats["question_types"]
                    )
                    stats["answer_types"] = add_dicts(stats["answer_types"], instance.stats["answer_types"])
                    stats["correct_answers"] = add_dicts(
                        stats["correct_answers"], instance.stats["correct_answers"]
                    )
                    stats["correct_answers"] = filter_dict_by_keys(
                        stats["correct_answers"], ["A", "B", "C", "A", "B", True, False]
                    )
                except InvalidPromptSequenceError:
                    attempts += 1
                    if attempts > max_attempts:
                        raise RuntimeError("Too many attempts to generate a valid prompt sequence.")

                    # redo the current iteration if Error is raised
                    stats["InvalidPromptSequenceErrors"] += 1
                else:
                    attempts = 0
                    index += 1
                    progress.update(sampling_task, advance=1)
    except KeyboardInterrupt:
        console.print("[red]Interrupted by user. Stopping generation but saving stats and plots...")

    console.print(
        f"[green]Finished Sampling {index} data instances (only the first {min(index, print_first)} were shown above)."
    )
    console.print("[yellow]Collecting stats and plotting...")

    # filters and sorts stats dict.
    stats["correct_answers"] = filter_dict_by_keys(
        stats["correct_answers"], ["A", "B", "C", "A", "B", True, False]
    )
    stats["question_types"] = {k: stats["question_types"][k] for k in sorted(stats["question_types"])}
    stats["answer_types"] = {k: stats["answer_types"][k] for k in sorted(stats["answer_types"])}

    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # plotting stats at the end of the generation process
    plt.figure(1)
    values_QT = list(stats["question_types"].values())
    labels_QT = list(stats["question_types"].keys())
    plt.barh(range(len(stats["question_types"])), values_QT)
    plt.yticks(range(len(stats["question_types"])), labels_QT)
    for i, v in enumerate(values_QT):
        plt.text(v, i, str(v), color="black", va="center")
    plt.savefig(output_dir / "question_types.png")

    plt.figure(2)
    values_AT = list(stats["answer_types"].values())
    labels_AT = list(stats["answer_types"].keys())
    plt.barh(range(len(stats["answer_types"])), values_AT)
    plt.yticks(range(len(stats["answer_types"])), labels_AT)
    for i, v in enumerate(values_AT):
        plt.text(v, i, str(v), color="black", va="center")
    plt.savefig(output_dir / "answer_types.png")

    plt.figure(3)
    values_CA = list(stats["correct_answers"].values())
    labels_CA = list(stats["correct_answers"].keys())
    plt.barh(range(len(stats["correct_answers"])), values_CA)
    plt.yticks(range(len(stats["correct_answers"])), labels_CA)
    for i, v in enumerate(values_CA):
        plt.text(v, i, str(v), color="black", va="center")
    plt.savefig(output_dir / "correct_answers.png")

    console.print(f"[green]Written everything to {output_dir}")
