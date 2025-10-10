from pathlib import Path

import numpy as np

from .....base.utility import (
    InvalidPromptSequenceError,
    count_actions_in_interval,
    create_options_dict,
    determine_multi_letter,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_multi_choice_answer_templates,
)
from ....question import Question


class IntervalQuestion(Question):
    """If a person performs the same action multiple times, this is counted as one action."""

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.interval_part_sequence_binary,
            1: self.interval_part_sequence_multi,
            2: self.interval_part_sequence_open,
            3: self.interval_whole_sequence_binary,
            4: self.interval_whole_sequence_multi,
            5: self.interval_whole_sequence_open,
        }

        self.templates_partial_binary = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_partial_binary.txt"
        )
        self.templates_partial_multi = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_partial_multi.txt"
        )
        self.templates_partial_open = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_partial_open.txt"
        )
        self.templates_whole_binary = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_whole_binary.txt"
        )
        self.templates_whole_multi = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_whole_multi.txt"
        )
        self.templates_whole_open = read_inputs(
            Path(__file__).parent / "descriptive_counting_interval_whole_open.txt"
        )

    def sample_question_answer_pair(
        self,
        actions: np.ndarray[str],
        time_stamps_end: np.ndarray[np.float64],
        answer_type: str,
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a sample question-answer pair based on the provided actions and time stamps.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.
            answer_type: Determines if all functions are eligible or only a certain one of the types open, multi or binary

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if answer_type == "all":
            random_int = self.rng.integers(0, 6).item()
        elif answer_type == "binary":
            random_int = self.rng.choice([0, 3]).item()
        elif answer_type == "multi":
            random_int = self.rng.choice([1, 4]).item()
        elif answer_type == "open":
            random_int = self.rng.choice([2, 5]).item()
        else:
            raise ValueError("no acceptable answer mode")
        return self.functions[random_int](actions, time_stamps_end)

    def sample_start_end(self, time_stamps_end: np.ndarray[np.float64]) -> tuple[float, float]:
        time1 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        time2 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        if time1 < time2:
            return time1, time2
        return time2, time1

    def interval_part_sequence_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair that validates if a specific number of actions are performed during a randomly chosen time interval

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        start, end = self.sample_start_end(time_stamps_end)

        correct = count_actions_in_interval(start, end, actions, time_stamps_end)
        if self.rng.choice((True, False)):
            question = (
                self.rng.choice(self.templates_partial_binary)
                .item()
                .format(number=correct, time1=start, time2=end)
            )
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            if actions.size <= 1:
                wrong = 2  # This is an arbitrary number that is not the correct answer
            else:
                wrong = self.rng.choice(list(set(range(1, actions.size + 1)) - set((correct,))))
            question = (
                self.rng.choice(self.templates_partial_binary)
                .item()
                .format(number=wrong, time1=start, time2=end)
            )
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"
        options_dict = create_options_dict("binary")
        return question, answer, "interval_part_sequence_binary", "binary", options_dict, label

    def interval_part_sequence_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair that gives 3 answer possibilities if a specific number of actions are performed during a randomly chosen time interval

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size <= 2:
            raise InvalidPromptSequenceError()
        start, end = self.sample_start_end(time_stamps_end)
        correct = count_actions_in_interval(start, end, actions, time_stamps_end)
        excludes = [correct]
        excludes.append(self.rng.choice(list(set(range(1, actions.size + 1)) - set(excludes))))
        excludes.append(self.rng.choice(list(set(range(1, actions.size + 1)) - set(excludes))))
        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", list(map(str, excludes)), indices)
        question = (
            self.rng.choice(self.templates_partial_multi)
            .item()
            .format(
                time1=start,
                time2=end,
                number1=excludes[indices[0]],
                number2=excludes[indices[1]],
                number3=excludes[indices[2]],
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct)
        return question, answer, "interval_part_sequence_multi", "multi", options_dict, label

    def interval_part_sequence_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict | None, str]:
        """
        Generates a qna pair that asks how many distinct actions are performed during a randomly chosen time interval

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        start, end = self.sample_start_end(time_stamps_end)
        correct = count_actions_in_interval(start, end, actions, time_stamps_end)
        question = self.rng.choice(self.templates_partial_open).item().format(time1=start, time2=end)
        answer = str(correct)
        return question, answer, "interval_part_sequence_open", "open", None, answer

    def interval_whole_sequence_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair that validates if a specific number of actions are performed during the whole time series

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        correct = len(set(actions))
        if self.rng.choice((True, False)):
            question = self.rng.choice(self.templates_whole_binary).item().format(number=correct)
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            if actions.size <= 1:
                wrong = 2  # This is an arbitrary number that is not the correct answer
            else:
                wrong = self.rng.choice(list(set(range(1, actions.size + 1)) - set((correct,))))
            question = self.rng.choice(self.templates_whole_binary).item().format(number=wrong)
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"
        options_dict = create_options_dict("binary")
        return question, answer, "interval_whole_sequence_binary", "binary", options_dict, label

    def interval_whole_sequence_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair that gives 3 answer possibilities if a specific number of actions are performed during the whole time series

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size <= 2:
            raise InvalidPromptSequenceError()
        correct = len(set(actions))
        excludes = [correct]
        excludes.append(self.rng.choice(list(set(range(1, actions.size + 1)) - set(excludes))))
        excludes.append(self.rng.choice(list(set(range(1, actions.size + 1)) - set(excludes))))
        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", list(map(str, excludes)), indices)
        question = (
            self.rng.choice(self.templates_whole_multi)
            .item()
            .format(
                number1=excludes[indices[0]],
                number2=excludes[indices[1]],
                number3=excludes[indices[2]],
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct)
        return question, answer, "interval_whole_sequence_multi", "multi", options_dict, label

    def interval_whole_sequence_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict | None, str]:
        """
        Generates a qna pair that asks how many distinct actions are performed during the whole time series

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        question = self.rng.choice(self.templates_whole_open).item()
        answer = str(len(set(actions)))
        return question, answer, "interval_whole_sequence_open", "open", None, answer
