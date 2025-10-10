from pathlib import Path

import numpy as np

from ....base.utility import (
    create_options_dict,
    determine_multi_letter,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
    sample_open_answer_templates,
    what_is_happening_at_t,
)
from ...question import Question


class DescriptiveIdentificationQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.descriptive_identification_binary,
            1: self.descriptive_identification_multi,
            2: self.descriptive_identification_open,
        }

        self.templates_identification_binary = read_inputs(
            Path(__file__).parent / "descriptive_identification_binary.txt"
        )
        self.templates_identification_multi = read_inputs(
            Path(__file__).parent / "descriptive_identification_multi.txt"
        )
        self.templates_identification_open = read_inputs(
            Path(__file__).parent / "descriptive_identification_open.txt"
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

    def descriptive_identification_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the validation of an action at a certain timestamp.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        # draw random float number in length of timeseries with 3 decimal digits
        time = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        real_activity = what_is_happening_at_t(time, actions, time_stamps_end)

        if self.rng.integers(0, 2) == 0:
            activity = real_activity
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
            label = "A"
        else:
            if actions.size == 1:
                false_activity = sample_deviating_actions(1, exclude=[real_activity], rng=self.rng).item()
            else:
                false_activity = self.rng.choice(actions[actions != real_activity])
            assert real_activity != false_activity
            activity = false_activity
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()
            label = "B"

        question = (
            self.rng.choice(self.templates_identification_binary).item().format(activity=activity, time=time)
        )
        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "descriptive_identification_binary", "binary", options_dict, label)
        return qna_pair

    def descriptive_identification_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the identification of an action at a certain timestamp given multiple answer possibilities.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        # draw random float number in length of timeseries with 3 decimal digits
        time = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        real_activity = what_is_happening_at_t(time, actions, time_stamps_end)

        all_false_options = list(set(actions[actions != real_activity]))
        if len(all_false_options) < 2:
            all_false_options.extend(
                sample_deviating_actions(2 - len(all_false_options), exclude=actions, rng=self.rng)
            )
        false_activities = self.rng.choice(all_false_options, 2, replace=False)
        assert real_activity not in false_activities

        choices = [real_activity, *false_activities]
        indices = self.rng.permutation(3)
        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)
        question = (
            self.rng.choice(self.templates_identification_multi)
            .item()
            .format(
                activity_1=choices[indices[0]],
                activity_2=choices[indices[1]],
                activity_3=choices[indices[2]],
                time=time,
            )
        )
        answer = sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=real_activity)
        qna_pair = (question, answer, "descriptive_identification_multi", "multi", options_dict, label)
        return qna_pair

    def descriptive_identification_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the identification of an action at a certain timestamp.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        # draw random float number in length of timeseries with 3 decimal digits
        time = round(self.rng.uniform(0, time_stamps_end[-1]), 3)
        real_activity = what_is_happening_at_t(time, actions, time_stamps_end)

        question = self.rng.choice(self.templates_identification_open).item().format(time=time)
        answer = sample_open_answer_templates(1, rng=self.rng).item().format(activity=real_activity)

        qna_pair = (question, answer, "descriptive_identification_open", "open", None, real_activity)
        return qna_pair
