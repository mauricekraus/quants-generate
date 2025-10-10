import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

DataInstanceSubclass = TypeVar("DataInstanceSubclass", bound="DataInstance")


@dataclass(frozen=False)
class DataInstance:
    """
    Represents a data instance containing information about a specific action sequence.

    Attributes:
        index: The index of the data instance.
        prompt_sequence: The sequence of prompts describing the actions.
        textual_description: The textual description of the action sequence.
        qa_pairs: The question-answer pairs related to the action sequence.
        trajectory: The trajectory data associated with the action sequence.
            It is an array of shape (batch_size, joint, xyz, timestamp).
        dataset_dir: The path to the dataset directory.
    """

    index: int
    actions: np.ndarray[str]
    prompt_sequence: Sequence[tuple[float, float, str]]
    textual_description: str
    qa_pairs: Sequence[tuple[str, str, str, str, dict, str]]
    stats: dict[str, dict[str, int]] | None = None
    trajectory: None | (NDArray[np.floating]) = None  # Of shape (batch_size joint xyz timestamp)

    dataset_dir: Path | None = None  # The path to the dataset

    @cached_property
    def instance_dir(self) -> Path:
        """
        Returns the directory path for the data instance.

        Returns:
            The directory path for the data instance.
        Raises:
            AssertionError: If the dataset directory is not set or does not exist.
        """
        assert self.dataset_dir is not None, "the dataset_dir must be set to a valid path"
        assert self.dataset_dir.exists(), f"the dataset_dir should already exist: {self.dataset_dir}"

        return self.dataset_dir / str(self.index)

    @property
    def total_length(self) -> float:
        """
        Calculate and return the total length of the data instance.

        Returns:
            The total length of the data instance in seconds.
        """
        last_prompt = self.prompt_sequence[-1]
        # cm: get time end of last_prompt (end of last action): _,xxx,_ (ignore, take, ignore)
        _, timestamp_end, _ = last_prompt
        return timestamp_end

    def to_disk(self) -> None:
        """
        Save and instance to disk by creating and overwriting the "data.json" and "trajectory.npy" files.

        Raises:
            OSError: If there is an error in creating or writing to the files.
        """

        self.instance_dir.mkdir(exist_ok=True, parents=False)

        # convert qa_pairs tuple sequence to dict sequence
        dict_qa_pairs = list()
        for qa_pair in self.qa_pairs:
            dict_qa_pairs.append(
                {
                    "question": qa_pair[0],
                    "answer": qa_pair[1],
                    "question_type": qa_pair[2],
                    "answer_type": qa_pair[3],
                    "options": qa_pair[4],
                    "correct_option": qa_pair[5],
                }
            )

        dict_prompt_sequence = list()
        for i, prompt in enumerate(self.prompt_sequence):
            dict_prompt_sequence.append(
                {
                    "start": prompt[0],
                    "end": prompt[1],
                    "action": self.actions[i],
                    "action_sentence": prompt[2],
                }
            )

        # Overwrites
        with open(self.instance_dir / "data.json", mode="w", encoding="utf-8") as data_file:
            content = {
                # the index is already in the path so we do not repeat it here
                "prompt_sequence": dict_prompt_sequence,
                "textual_description": self.textual_description,
                "qa_pairs": dict_qa_pairs,
            }
            json.dump(content, data_file, ensure_ascii=False, indent=4)

        if self.trajectory is not None:
            # Overwrites
            # Note: It is not worth it to use np.savez_compressed here since the
            # gains are minimal (only ~10% smaller file size) and the
            # decompression time would increase (not benchmarked, though).
            # (Tested with `batch_size=1`.)
            np.save(
                self.instance_dir / "trajectory.npy",
                self.trajectory,
                allow_pickle=False,
                fix_imports=False,
            )

    @classmethod
    def from_disk(cls: type[DataInstanceSubclass], dataset_dir: Path, index: int) -> "DataInstanceSubclass":
        """
        Load an instance from disk.

        Args:
            cls: The class of the DataInstanceSub object.
            dataset_dir: The directory where the dataset is stored.
            index: The index of the data instance to load.

        Returns:
            The loaded DataInstance object.

        Raises:
            AssertionError: If the dataset directory or instance folder does not exist.

        """
        assert dataset_dir.exists(), f"the dataset_dir must exist: {dataset_dir}"
        instance_dir = dataset_dir / str(index)
        assert instance_dir.exists(), f"the folder of the instance does not exist: {instance_dir}"

        with open(instance_dir / "data.json", encoding="utf-8") as data_file:
            data = json.load(data_file)

        trajectory_file = instance_dir / "trajectory.npy"
        if trajectory_file.exists():
            trajectory = np.load(
                trajectory_file,
                allow_pickle=False,
                fix_imports=False,
            )
        else:
            trajectory = None

        return cls(
            index=index,
            prompt_sequence=list(map(tuple, data["prompt_sequence"])),
            textual_description=data["textual_description"],
            qa_pairs=list(map(tuple, data["qa_pairs"])),
            trajectory=trajectory,
            dataset_dir=dataset_dir,
        )

    def print(self, to: Console | None) -> None:
        """
        Prints the details of a data instance. In this case used to print the question/answer pairs together with the corresponding actions and timestamps to the console in a pretty table.

        Args:
            to: The console to use for printing. Defaults to a newly created one.

        """
        table = Table(title=f"Data instance {self.index:06d}", width=100)
        table.add_column("start", justify="right", style="cyan", ratio=0.1)
        table.add_column("end", justify="right", style="magenta", ratio=0.1)
        table.add_column("prompt sequence", style="green", ratio=0.8)
        for timestamp_start, timestamp_end, prompt in self.prompt_sequence:
            table.add_row(f"{timestamp_start:.4f}", f"{timestamp_end:.4f}", prompt)
        table.caption = "\n\n".join(
            f"Q: {question}\n A: {answer}\n QCat: {q_cat}\n ACat: {a_cat}\n Opt: {options}\n Truth: {truth}"
            for question, answer, q_cat, a_cat, options, truth in self.qa_pairs
        )

        if to is None:
            to = Console()
        to.print(table)
