from collections import defaultdict
from math import ceil
from pathlib import Path

import numpy as np

from .....base.utility import (
    InvalidPromptSequenceError,
    count_occurrences,
    create_options_dict,
    determine_multi_letter,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
)
from ....question import Question


class ActionCountQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.action_count_binary,
            1: self.action_count_multi,
            2: self.action_count_number_multi,
            3: self.action_count_number_open,
            4: self.action_count_open,
        }

        self.templates_action_count_open = read_inputs(
            Path(__file__).parent / "descriptive_action_counting_open.txt"
        )
        self.templates_action_count_binary = read_inputs(
            Path(__file__).parent / "descriptive_action_counting_binary.txt"
        )
        self.templates_action_count_multi = read_inputs(
            Path(__file__).parent / "descriptive_action_counting_multi.txt"
        )
        self.templates_action_count_number_open = read_inputs(
            Path(__file__).parent / "descriptive_action_counting_number_open.txt"
        )
        self.templates_action_count_number_multi = read_inputs(
            Path(__file__).parent / "descriptive_action_counting_number_multi.txt"
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
            random_int = self.rng.integers(0, 5).item()
        elif answer_type == "binary":
            random_int = 0
        elif answer_type == "multi":
            random_int = self.rng.choice([1, 2]).item()
        elif answer_type == "open":
            random_int = self.rng.choice([3, 4]).item()
        else:
            raise ValueError("no acceptable answer mode")
        return self.functions[random_int](actions, time_stamps_end)

    def action_with_occ_and_negatives(
        self,
        actions: np.ndarray[str],
        time_stamps_end: np.ndarray[np.float64],
        min_wrong_occurences: int = 0,
        min_wrong_actions: int = 0,
    ) -> tuple[str, int, list[int]]:
        if self.rng.integers(0, 2).item() == 0:
            index = self.rng.integers(0, actions.size).item()
            action = actions[index]
        else:
            action = sample_deviating_actions(1, exclude=list(set(actions)), rng=self.rng).item()

        occ = count_occurrences(action, actions)

        max_occurences = int(ceil(time_stamps_end.size / 2))
        wrong_occurences = [i for i in range(0, max_occurences + 1) if i != occ]
        # If that is not enough, we add some more unreasonable answers
        num_missing = min_wrong_occurences - len(wrong_occurences)
        if num_missing > 0:
            wrong_occurences.extend(range(max_occurences + 2, max_occurences + 2 + num_missing))

        ambiguous_actions = [action for action in set(actions) if count_occurrences(action, actions) == occ]
        wrong_actions = [action for action in set(actions) if action not in ambiguous_actions]
        # If that is not enough, we add some more unreasonable answers
        num_missing = min_wrong_actions - len(wrong_actions)
        if num_missing > 0:
            if occ == 0:
                # Where should we get them from?
                raise InvalidPromptSequenceError()

            wrong_actions.extend(
                sample_deviating_actions(num_missing, exclude=actions, rng=self.rng).tolist()
            )

        return action, occ, wrong_occurences, wrong_actions

    def action_count_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the validation of the count of an action during the timeseries.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        action, occ, wrong_answers, _ = self.action_with_occ_and_negatives(
            actions, time_stamps_end, min_wrong_occurences=1
        )

        if self.rng.integers(0, 2).item() == 0 or len(set(actions)) == 1:
            question = (
                self.rng.choice(self.templates_action_count_binary).item().format(activity=action, number=occ)
            )
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            false_occ = self.rng.choice(wrong_answers).item()
            question = (
                self.rng.choice(self.templates_action_count_binary)
                .item()
                .format(activity=action, number=false_occ)
            )
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "count_binary", "binary", options_dict, label)
        return qna_pair

    def action_count_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on multiple answer possibilities of the count of an action during the timeseries.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        action, occ, wrong_answers, _ = self.action_with_occ_and_negatives(
            actions, time_stamps_end, min_wrong_occurences=2
        )

        options = [occ, *self.rng.choice(wrong_answers, 2, replace=False).tolist()]
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        question = (
            self.rng.choice(self.templates_action_count_multi)
            .item()
            .format(
                activity=action,
                number1=options[indices[0]],
                number2=options[indices[1]],
                number3=options[indices[2]],
            )
        )

        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=occ)
        options_dict = create_options_dict("multi", options, indices)

        qna_pair = (question, answer, "count_multi", "multi", options_dict, label)
        return qna_pair

    def action_count_number_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the identification of an action taht occurs n times during the timeseries with given answer possibilities.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        action, occ, _, wrong_answers = self.action_with_occ_and_negatives(
            actions, time_stamps_end, min_wrong_actions=2
        )

        choices = [action, *self.rng.choice(wrong_answers, 2, replace=False).tolist()]
        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)

        question = (
            self.rng.choice(self.templates_action_count_number_multi)
            .item()
            .format(
                number=occ,
                activity1=choices[indices[0]],
                activity2=choices[indices[1]],
                activity3=choices[indices[2]],
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=action)
        options_dict = create_options_dict("multi", choices, indices)

        qna_pair = (question, answer, "count_number_multi", "multi", options_dict, label)
        return qna_pair

    def action_count_number_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the identification of an action that occurs n times during the timeseries.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        counts_to_actions = defaultdict(list)
        for action in actions:
            counts_to_actions[count_occurrences(action, actions)].append(action)
        occ = self.rng.choice(list(counts_to_actions.keys()))
        ambiguities = list(set(counts_to_actions[occ]))
        self.rng.shuffle(ambiguities)

        question = self.rng.choice(self.templates_action_count_number_open).item().format(number=occ)
        answer = ", ".join(ambiguities)

        qna_pair = (question, answer, "count_number_open", "open", None, answer)
        return qna_pair

    def action_count_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the counting of the occurrences of an action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """

        # Due to the way the question is phrased, we can only ask for the count of an action that actually occurs

        activity = self.rng.choice(actions)
        count = count_occurrences(activity, actions)

        question = self.rng.choice(self.templates_action_count_open).item().format(activity=activity)
        answer = str(count)

        qna_pair = (question, answer, "count_open", "open", None, answer)
        return qna_pair
