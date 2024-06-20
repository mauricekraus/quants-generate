from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ....base.utility import (
    InvalidPromptSequenceError,
    create_options_dict,
    determine_multi_letter,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
)
from ...question import Question


class ComparisonCounting(Question):
    """It is important to note that conscutive actions are treated as one continous action."""

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.comparison_counting_binary,
            1: self.comparison_counting_multi,
            2: self.comparison_counting_open,
        }

        self.templates_comparison_counting_binary = read_inputs(
            Path(__file__).parent / "comparison_counting_binary.txt"
        )
        self.templates_comparison_counting_multi = read_inputs(
            Path(__file__).parent / "comparison_counting_multi.txt"
        )
        self.templates_comparison_counting_open = read_inputs(
            Path(__file__).parent / "comparison_counting_open.txt"
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

    def _pair_of_maybe_same_frequency(
        self, actions: np.ndarray[str], rng: np.random.Generator
    ) -> tuple[str, str]:
        """
        Generate a pair of actions that have, with best-effort "50% chance" of having the same frequency.

        Args:
            actions: A sequence of actions.

        Returns:
            A tuple containing the two actions.
        """

        if actions.size < 2:
            raise InvalidPromptSequenceError()

        counts_to_actions = defaultdict(list)
        for key, val in Counter(actions).items():
            counts_to_actions[val].append(key)

        # Try to find a pair with the same frequency
        # Oversample higher frequencies
        counts_to_actions_multi = {
            count: action for count, action in counts_to_actions.items() if len(action) > 1
        }

        same_possible = bool(counts_to_actions_multi)
        different_possible = len(counts_to_actions) > 1

        balance_factor = 0  # Currently, we always try to generate a pair with the same frequency
        if rng.random() < balance_factor and same_possible or not different_possible:
            # Case 1: same frequencies
            p = np.array([len(v) for v in counts_to_actions_multi.values()])
            freq = rng.choice(list(counts_to_actions_multi.keys()), p=p / p.sum())
            return rng.choice(counts_to_actions_multi[freq], 2, replace=False)
        else:
            # Case 2: different frequencies
            freqs = list(counts_to_actions.keys())
            freq_1, freq_2 = rng.choice(freqs, 2, replace=False)
            return rng.choice(counts_to_actions[freq_1]), rng.choice(counts_to_actions[freq_2])

    def comparison_counting_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the evaluation of the amount of action occurrences.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        action_counts = Counter(actions)

        action_1, action_2 = self._pair_of_maybe_same_frequency(actions, self.rng)

        if action_counts[action_1] == action_counts[action_2]:
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        question = (
            self.rng.choice(self.templates_comparison_counting_binary)
            .item()
            .format(activity_1=action_1, activity_2=action_2)
        )
        options_dict = create_options_dict("binary")

        qna_pair = (question, answer, "comparison_counting_binary", "binary", options_dict, label)
        return qna_pair

    def _multiple_choice_same_frequency(
        self,
        actions: np.ndarray[str],
        required_wrong_answers: int,
        enforce_unique_pair: bool = False,
    ) -> tuple[str, str, list[str]]:
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        counts_to_actions = defaultdict(list)
        for key, val in Counter(actions).items():
            counts_to_actions[val].append(key)

        # Try to find a pair with the same frequency
        counts_to_actions_multi = {
            count: actions
            for count, actions in counts_to_actions.items()
            if len(actions) > 1 and (not enforce_unique_pair or len(actions) == 2)
        }

        if not counts_to_actions_multi:
            raise InvalidPromptSequenceError("No two actions with the same frequency found.")

        # Find a pair of actions with the same frequency
        count = self.rng.choice(list(counts_to_actions_multi.keys()))
        positive_actions = counts_to_actions_multi[count]
        activity_anchor, correct_activity = self.rng.choice(positive_actions, 2, replace=False)
        negative_actions = list(set(action for action in actions if action not in positive_actions))

        # Sample two wrong answers
        wrong_answers = self.rng.choice(
            negative_actions, min(required_wrong_answers, len(negative_actions)), replace=False
        ).tolist()
        if len(wrong_answers) < required_wrong_answers:
            # We need to sample more wrong answers from outside the visible actions
            wrong_answers.extend(
                sample_deviating_actions(2 - len(wrong_answers), exclude=actions, rng=self.rng)
            )

        return activity_anchor, correct_activity, wrong_answers

    def comparison_counting_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the evaluating of 3 answer possibilities of possible amount of action occurrences.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        activity_anchor, correct_activity, wrong_answers = self._multiple_choice_same_frequency(
            actions, required_wrong_answers=2, enforce_unique_pair=False
        )

        choices = [correct_activity, *wrong_answers]
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_comparison_counting_multi)
            .item()
            .format(
                activity=activity_anchor,
                activity_1=choices[indices[0]],
                activity_2=choices[indices[1]],
                activity_3=choices[indices[2]],
            )
        )
        answer = (
            sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct_activity)
        )

        qna_pair = (question, answer, "comparison_counting_multi", "multi", options_dict, label)
        return qna_pair

    def comparison_counting_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the evaluating of action occurrences.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        activity_anchor, correct_activity, _ = self._multiple_choice_same_frequency(
            actions, required_wrong_answers=0, enforce_unique_pair=True
        )

        question = (
            self.rng.choice(self.templates_comparison_counting_open).item().format(activity=activity_anchor)
        )
        answer = (
            sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=correct_activity)
        )

        qna_pair = (question, answer, "comparison_counting_open", "open", None, correct_activity)
        return qna_pair
