from pathlib import Path

import numpy as np

from ....base.utility import (
    InvalidPromptSequenceError,
    create_options_dict,
    is_synonymous,
    is_unique,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    what_is_happening_at_t,
)
from ...question import Question


class ComparisonAction(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.action_comparison_binary,
            1: self.action_comparison_first_last_different_binary,
            2: self.action_comparison_first_last_same_binary,
            3: self.action_comparison_timestamp_different_binary,
            4: self.action_comparison_timestamp_same_binary,
        }

        self.templates_comparison_first_last_same_binary = read_inputs(
            Path(__file__).parent / "comparison_action_first_last_same_binary.txt"
        )
        self.templates_comparison_first_last_different_binary = read_inputs(
            Path(__file__).parent / "comparison_action_first_last_different_binary.txt"
        )
        self.templates_comparison_action_binary = read_inputs(
            Path(__file__).parent / "comparison_action_binary.txt"
        )
        self.templates_comparison_action_timestamp_same_binary = read_inputs(
            Path(__file__).parent / "comparison_action_timestamp_same_binary.txt"
        )
        self.templates_comparison_action_timestamp_different_binary = read_inputs(
            Path(__file__).parent / "comparison_action_timestamp_different_binary.txt"
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
        if answer_type == "all" or answer_type == "binary":
            random_int = self.rng.integers(0, 5).item()
        elif answer_type == "multi" or answer_type == "open":
            raise InvalidPromptSequenceError()
        else:
            raise ValueError("no acceptable answer mode")
        return self.functions[random_int](actions, time_stamps_end)

    def action_comparison_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the actions before and after a random action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 3:
            raise InvalidPromptSequenceError()

        action_index = self.rng.integers(1, actions.size - 1).item()
        if not is_unique(action_index, actions):
            raise InvalidPromptSequenceError()

        if is_synonymous(actions[action_index - 1], actions[action_index + 1]):
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        options_dict = create_options_dict("binary")
        question = (
            self.rng.choice(self.templates_comparison_action_binary)
            .item()
            .format(activity=actions[action_index])
        )

        qna_pair = (question, answer, "comparison_binary", "binary", options_dict, label)
        return qna_pair

    def action_comparison_first_last_different_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the first and last actions. True if first and last are different.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        first_action = actions[0]
        last_action = actions[-1]
        question = self.rng.choice(self.templates_comparison_first_last_different_binary).item()

        if not is_synonymous(first_action, last_action):
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "comparison_first_last_different_binary", "binary", options_dict, label)

        return qna_pair

    def action_comparison_first_last_same_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the first and last actions. True if first and last are same.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        first_action = actions[0]
        last_action = actions[-1]
        question = self.rng.choice(self.templates_comparison_first_last_same_binary).item()

        if is_synonymous(first_action, last_action):
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "comparison_first_last_same_binary", "binary", options_dict, label)
        return qna_pair

    def action_comparison_timestamp_different_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the actions happening at two random times. True if they are same.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        time_1 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        time_2 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        question = (
            self.rng.choice(self.templates_comparison_action_timestamp_different_binary)
            .item()
            .format(time_1=time_1, time_2=time_2)
        )
        activity_1 = what_is_happening_at_t(time_1, actions, time_stamps_end)
        activity_2 = what_is_happening_at_t(time_2, actions, time_stamps_end)
        if not is_synonymous(activity_1, activity_2):
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"
        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "comparison_timestamp_different_binary", "binary", options_dict, label)
        return qna_pair

    def action_comparison_timestamp_same_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the first and last actions. True if first and last are same.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        time_1 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        time_2 = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        question = (
            self.rng.choice(self.templates_comparison_action_timestamp_same_binary)
            .item()
            .format(time_1=time_1, time_2=time_2)
        )
        activity_1 = what_is_happening_at_t(time_1, actions, time_stamps_end)
        activity_2 = what_is_happening_at_t(time_2, actions, time_stamps_end)
        if is_synonymous(activity_1, activity_2):
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"
        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "comparison_timestamp_same_binary", "binary", options_dict, label)
        return qna_pair
