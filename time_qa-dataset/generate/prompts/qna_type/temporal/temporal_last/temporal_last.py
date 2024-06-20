from pathlib import Path

import numpy as np

from ....base.utility import (
    binary_no_answer_templates,
    binary_yes_answer_templates,
    create_options_dict,
    determine_multi_letter,
    open_answer_templates,
    read_inputs,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
)
from ...question import Question


class LastQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.last_binary,
            1: self.last_multi,
            2: self.last_open,
        }

        self.templates_last_open = read_inputs(Path(__file__).parent / "temporal_last_open.txt")
        self.templates_last_binary = read_inputs(Path(__file__).parent / "temporal_last_binary.txt")
        self.templates_last_multi = read_inputs(Path(__file__).parent / "temporal_last_multi.txt")

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
            random_int = self.rng.integers(0, 3).item()
        elif answer_type == "binary":
            random_int = 0
        elif answer_type == "multi":
            random_int = 1
        elif answer_type == "open":
            random_int = 2
        else:
            raise ValueError("no acceptable answer mode")
        return self.functions[random_int](actions, time_stamps_end)

    def last_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a binary question-answer pair based on the last activity in the list. Checks is the chosen activity is the last activity.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        last_activity = actions[-1]
        other_activities = actions[:-1]

        if actions.size == 1 or self.rng.integers(0, 2).item() == 0:
            activity = last_activity
            label = "A"
            answer = self.rng.choice(binary_yes_answer_templates).item()
        else:
            activity = self.rng.choice(other_activities).item()
            label = "B"
            answer = self.rng.choice(binary_no_answer_templates).item()

        question = self.rng.choice(self.templates_last_binary).item().format(activity=activity)
        options_dict = create_options_dict("binary")

        qna_pair = (question, answer, "last_binary", "binary", options_dict, label)
        return qna_pair

    def last_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question and answer pair for the last activity in a sequence of actions. Gives multi options and chosses the right one.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        last_activity = actions[-1]

        wrong_choices = sample_deviating_actions(2, exclude=[last_activity], rng=self.rng)
        choices = [last_activity, *wrong_choices]
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_last_multi)
            .item()
            .format(
                activity_1=choices[indices[0]],
                activity_2=choices[indices[1]],
                activity_3=choices[indices[2]],
            )
        )

        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=last_activity)

        qna_pair = (question, answer, "last_multi", "multi", options_dict, label)
        return qna_pair

    def last_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question and answer pair for the last activity in a sequence of actions with an open-ended answer.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        last_activity = actions[len(actions) - 1]

        question = self.rng.choice(self.templates_last_open).item()
        answer = self.rng.choice(open_answer_templates).item().format(activity=last_activity)

        qna_pair = (question, answer, "last_open", "open", None, last_activity)
        return qna_pair
