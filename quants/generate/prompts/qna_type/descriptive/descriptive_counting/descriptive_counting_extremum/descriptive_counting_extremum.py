from collections import defaultdict
from pathlib import Path

import numpy as np

from .....base.utility import (
    InvalidPromptSequenceError,
    count_occurrences,
    create_options_dict,
    determine_multi_letter,
    format_list_to_string,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
)
from ....question import Question


class ExtremumQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.extremum_least_binary,
            1: self.extremum_least_multi,
            2: self.extremum_least_open,
            3: self.extremum_most_binary,
            4: self.extremum_most_multi,
            5: self.extremum_most_open,
        }

        self.templates_extremum_least_open = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_least_open.txt"
        )
        self.templates_extremum_most_open = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_most_open.txt"
        )
        self.templates_extremum_least_multi = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_least_multi.txt"
        )
        self.templates_extremum_most_multi = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_most_multi.txt"
        )
        self.templates_extremum_least_binary = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_least_binary.txt"
        )
        self.templates_extremum_most_binary = read_inputs(
            Path(__file__).parent / "descriptive_counting_extremum_most_binary.txt"
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

    def make_stats(self, actions: np.ndarray[str]) -> tuple[int, int, list[str], list[str]]:
        occurences = {action: count_occurrences(action, actions) for action in actions}

        actions_per_occurence = defaultdict(list)
        for action, occurence in occurences.items():
            actions_per_occurence[occurence].append(action)

        least = min(actions_per_occurence.keys())
        most = max(actions_per_occurence.keys())

        return least, most, actions_per_occurence[least], actions_per_occurence[most]

    def extremum_least_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a QnA pair that validates if a randomly chosen action is the least often performed.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        _, _, actions_least, _ = self.make_stats(actions)
        wrong_choices = list(set(actions) - set(actions_least))
        wrong_possible = bool(wrong_choices)  # if empty, wrong answer is not possible

        if not wrong_possible or self.rng.integers(0, 2).item() == 0:
            question = (
                self.rng.choice(self.templates_extremum_least_binary)
                .item()
                .format(activity=self.rng.choice(actions_least))
            )
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            question = (
                self.rng.choice(self.templates_extremum_least_binary)
                .item()
                .format(activity=self.rng.choice(wrong_choices))
            )
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "extremum_least_binary", "binary", options_dict, label)
        return qna_pair

    def extremum_least_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair with 3 answer possibilities assessing which action is performed the least often with excactly one correct answer.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        _, _, least_actions, _ = self.make_stats(actions)

        correct_answer = self.rng.choice(least_actions)
        wrong_answer_pool = list(set(actions) - set(least_actions))
        if len(wrong_answer_pool) < 2:
            # We can't sample ones that are not in the actions since they would have occurence zero
            # With four actions, this effectively triggers each time!!!
            raise InvalidPromptSequenceError()
        wrong_answers = sample_deviating_actions(2, actions, rng=self.rng).tolist()

        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)

        possibilities = [correct_answer, *wrong_answers]
        options_dict = create_options_dict("multi", possibilities, indices)

        question = (
            self.rng.choice(self.templates_extremum_least_multi)
            .item()
            .format(
                activity_1=possibilities[indices[0]],
                activity_2=possibilities[indices[1]],
                activity_3=possibilities[indices[2]],
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct_answer)
        qna_pair = (question, answer, "extremum_least_multi", "multi", options_dict, label)
        return qna_pair

    def extremum_least_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a QnA pair that asks which action is the least often performed

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        _, _, least_common_actions, _ = self.make_stats(actions)

        question = self.rng.choice(self.templates_extremum_least_open).item()
        answer = format_list_to_string(least_common_actions)

        qna_pair = (question, answer, "extremum_least_open", "open", None, answer)
        return qna_pair

    def extremum_most_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a QnA pair that validates if a randomly chosen action is the most often performed

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        _, _, _, actions_most = self.make_stats(actions)
        wrong_choices = list(set(actions) - set(actions_most))

        if self.rng.integers(0, 2).item() == 0:
            action = self.rng.choice(actions_most)
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            if wrong_choices:
                action = self.rng.choice(wrong_choices)
            else:
                action = sample_deviating_actions(1, exclude=actions, rng=self.rng).item()
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        question = self.rng.choice(self.templates_extremum_most_binary).item().format(activity=action)
        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "extremum_most_binary", "binary", options_dict, label)
        return qna_pair

    def extremum_most_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a qna pair with 3 answer possibilities assessing which action is performed the most often with excactly one correct answer.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        _, _, _, most_actions = self.make_stats(actions)

        correct_answer = self.rng.choice(most_actions)
        wrong_answer_pool = list(set(actions) - set(most_actions))
        if len(wrong_answer_pool) < 2:
            wrong_answer_pool.extend(
                sample_deviating_actions(2 - len(wrong_answer_pool), exclude=actions, rng=self.rng)
            )
        wrong_answers = sample_deviating_actions(2, actions, rng=self.rng).tolist()

        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)

        possibilities = [correct_answer, *wrong_answers]
        options_dict = create_options_dict("multi", possibilities, indices)

        question = (
            self.rng.choice(self.templates_extremum_most_multi)
            .item()
            .format(
                activity_1=possibilities[indices[0]],
                activity_2=possibilities[indices[1]],
                activity_3=possibilities[indices[2]],
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct_answer)
        qna_pair = (question, answer, "extremum_most_multi", "multi", options_dict, label)
        return qna_pair

    def extremum_most_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a QnA pair that asks which action is the most often performed

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        _, _, _, most_common_actions = self.make_stats(actions)

        question = self.rng.choice(self.templates_extremum_most_open).item()
        answer = format_list_to_string(most_common_actions)

        qna_pair = (question, answer, "extremum_most_open", "open", None, answer)
        return qna_pair
