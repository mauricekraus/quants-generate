from pathlib import Path

import numpy as np

from ....base.utility import (
    InvalidPromptSequenceError,
    create_options_dict,
    determine_multi_letter,
    format_list_to_string,
    is_unique,
    read_inputs,
    sample_binary_no_answer_templates,
    sample_binary_yes_answer_templates,
    sample_deviating_actions,
    sample_multi_choice_answer_templates,
    sample_open_answer_templates,
)
from ...question import Question


class AfterQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.after_binary,
            1: self.after_multi,
            2: self.after_open,
            3: self.right_after_binary,
            4: self.right_after_multi,
            5: self.right_after_open,
        }

        self.templates_happened_right_after_open = read_inputs(
            Path(__file__).parent / "temporal_after_directly_open.txt"
        )
        self.templates_happened_right_after_binary = read_inputs(
            Path(__file__).parent / "temporal_after_directly_binary.txt"
        )
        self.templates_happened_right_after_multi = read_inputs(
            Path(__file__).parent / "temporal_after_directly_multi.txt"
        )
        self.templates_after_binary = read_inputs(Path(__file__).parent / "temporal_after_binary.txt")
        self.templates_after_open = read_inputs(Path(__file__).parent / "temporal_after_open.txt")
        self.templates_after_multi = read_inputs(Path(__file__).parent / "temporal_after_multi.txt")

    def sample_question_answer_pair(
        self,
        actions: np.ndarray[str],
        time_stamps_end: np.ndarray[np.float64],
        answer_type: str,
    ) -> tuple[str, str, str, str, dict, str]:
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

    def after_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of activities after a given action.True if the drawn action appears in the actions following the given action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(0, len(actions) - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()
        activity = actions[index_anchor]
        activities_after = actions[index_anchor + 1 :]

        if self.rng.integers(0, 2).item() == 0:
            activity_2 = self.rng.choice(activities_after).item()
            label = "A"
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()

        else:
            activity_2 = sample_deviating_actions(1, exclude=activities_after, rng=self.rng).item()
            label = "B"
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()

        question = (
            self.rng.choice(self.templates_after_binary)
            .item()
            .format(activity_1=activity, activity_2=activity_2)
        )

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "after_binary", "binary", options_dict, label)
        return qna_pair

    def after_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a question and answer pair related to activities after a chosen activity. Multi choice answer is provieded if the chosen array before the chosen action is containing the correct after activities or not.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(0, len(actions) - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()
        chosen_activity = actions[index_anchor]
        after_activities = actions[index_anchor + 1 :]
        correct_after_activity = self.rng.choice(after_activities).item()

        false_activities = sample_deviating_actions(2, exclude=after_activities, rng=self.rng).tolist()

        choices = [correct_after_activity, *false_activities]
        assert len(set(choices)) == len(choices)
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_after_multi)
            .item()
            .format(
                activity=chosen_activity,
                activity_1=choices[indices[0]],
                activity_2=choices[indices[1]],
                activity_3=choices[indices[2]],
            )
        )

        answer = (
            sample_multi_choice_answer_templates(1, rng=self.rng)
            .item()
            .format(solution=correct_after_activity)
        )

        qna_pair = (question, answer, "after_multi", "multi", options_dict, label)
        return qna_pair

    def after_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of activities after a given action. The returned answer contains all actions following the chosen question

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        assert len
        if len(actions) == 1:
            raise InvalidPromptSequenceError("The actions array only contains one element.")
        index_anchor = self.rng.integers(0, len(actions) - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()
        anchor_action = actions[index_anchor]
        actions_after_anchor = actions[index_anchor + 1 :]

        # concatenate the actions into a nice string (e.g., "walking, jumping, and rope skipping")
        actions_after_anchor = format_list_to_string(actions_after_anchor)

        question = self.rng.choice(self.templates_after_open).item().format(activity=anchor_action)
        answer = sample_open_answer_templates(1, rng=self.rng).item().format(activity=actions_after_anchor)

        qna_pair = (question, answer, "after_open", "open", None, actions_after_anchor)
        return qna_pair

    def right_after_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the action after a given action. True if the drawn action is the following action of the given action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if len(actions) < 2:
            raise InvalidPromptSequenceError("The actions array requires at least two elements")

        index_anchor = self.rng.integers(0, len(actions) - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        activity = actions[index_anchor]
        activity_right_after = actions[index_anchor + 1]

        if self.rng.integers(0, 2).item() == 0:
            activity_2 = activity_right_after
            label = "A"
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
        else:
            activity_2 = sample_deviating_actions(
                1, exclude=[activity, activity_right_after], rng=self.rng
            ).item()
            label = "B"
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()

        question = (
            self.rng.choice(self.templates_happened_right_after_binary)
            .item()
            .format(activity_1=activity, activity_2=activity_2)
        )

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "right_after_binary", "binary", options_dict, label)
        return qna_pair

    def right_after_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the activity after the given action. The returned answer contains the action following the chosen action

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(0, actions.size - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        question = (
            self.rng.choice(self.templates_happened_right_after_open)
            .item()
            .format(activity=actions[index_anchor])
        )
        answer = (
            sample_open_answer_templates(1, rng=self.rng).item().format(activity=actions[index_anchor + 1])
        )

        return question, answer, "right_after_open", "open", None, actions[index_anchor + 1]

    def right_after_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a question and answer pair related to the activity after a chosen activity. Multi choice answer is provided if the activity after the chosen activity is the following one or not.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(0, actions.size - 1).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        chosen_activity = actions[index_anchor]
        right_after_activity = actions[index_anchor + 1]

        false_activities = sample_deviating_actions(
            2, exclude=[chosen_activity, right_after_activity], rng=self.rng
        ).tolist()

        choices = [right_after_activity, *false_activities]
        assert len(set(choices)) == len(choices)
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_happened_right_after_multi)
            .item()
            .format(
                activity=chosen_activity,
                activity_1=choices[indices[0]],
                activity_2=choices[indices[1]],
                activity_3=choices[indices[2]],
            )
        )

        answer = (
            sample_multi_choice_answer_templates(1, rng=self.rng).item().format(solution=right_after_activity)
        )

        return question, answer, "right_after_multi", "multi", options_dict, label
