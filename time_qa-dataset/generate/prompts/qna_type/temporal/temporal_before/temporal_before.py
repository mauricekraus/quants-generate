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


class BeforeQuestion(Question):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)

        self.functions = {
            0: self.before_binary,
            1: self.before_multi,
            2: self.before_open,
            3: self.right_before_binary,
            4: self.right_before_multi,
            5: self.right_before_open,
        }

        self.templates_happened_right_before_open = read_inputs(
            Path(__file__).parent / "temporal_before_directly_open.txt"
        )
        self.templates_happened_right_before_binary = read_inputs(
            Path(__file__).parent / "temporal_before_directly_binary.txt"
        )
        self.templates_happened_right_before_multi = read_inputs(
            Path(__file__).parent / "temporal_before_directly_multi.txt"
        )
        self.templates_before_binary = read_inputs(Path(__file__).parent / "temporal_before_binary.txt")
        self.templates_before_open = read_inputs(Path(__file__).parent / "temporal_before_open.txt")
        self.templates_before_multi = read_inputs(Path(__file__).parent / "temporal_before_multi.txt")

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

    def before_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of activities before a given action.True if the drawn action appears in the actions before the given action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(1, len(actions)).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()
        activity = actions[index_anchor]
        activities_before = actions[:index_anchor]

        if self.rng.integers(0, 2).item() == 0:
            activity_2 = self.rng.choice(activities_before).item()
            label = "A"
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()

        else:
            activity_2 = sample_deviating_actions(1, exclude=activities_before, rng=self.rng).item()
            label = "B"
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()

        question = (
            self.rng.choice(self.templates_before_binary)
            .item()
            .format(activity_1=activity, activity_2=activity_2)
        )

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "before_binary", "binary", options_dict, label)
        return qna_pair

    def before_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a question and answer pair related to activities before a chosen activity. Multi choice answer is provieded if the chosen array before the chosen action is containing the correct before activities or not.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(1, len(actions)).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        chosen_activity = actions[index_anchor]
        before_activities = actions[:index_anchor]
        correct_before_activity = self.rng.choice(before_activities).item()

        false_activities = sample_deviating_actions(2, exclude=before_activities, rng=self.rng).tolist()

        choices = [correct_before_activity, *false_activities]
        assert len(set(choices)) == len(choices)
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_before_multi)
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
            .format(solution=correct_before_activity)
        )

        qna_pair = (question, answer, "before_multi", "multi", options_dict, label)
        return qna_pair

    def before_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of activities before a given action. The returned answer contains all actions proceeding the chosen question

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size <= 2:
            raise InvalidPromptSequenceError()
        index_anchor = self.rng.integers(1, len(actions)).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()
        anchor_action = actions[index_anchor]
        actions_before_anchor = actions[:index_anchor]

        # concatenate the actions into a nice string (e.g., "walking, jumping, and rope skipping")
        result = format_list_to_string(actions_before_anchor)

        question = self.rng.choice(self.templates_before_open).item().format(activity=anchor_action)
        answer = sample_open_answer_templates(1, rng=self.rng).item().format(activity=result)

        qna_pair = (question, answer, "before_open", "open", None, result)
        return qna_pair

    def right_before_binary(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the action before a given action. True if the drawn action is the proceeding action of the given action.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if len(actions) < 2:
            raise InvalidPromptSequenceError("The actions array requires at least two elements")

        index_anchor = self.rng.integers(1, len(actions)).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        activity = actions[index_anchor]
        activity_right_before = actions[index_anchor - 1]

        if self.rng.integers(0, 2).item() == 0:
            activity_2 = activity_right_before
            label = "A"
            answer = sample_binary_yes_answer_templates(1, rng=self.rng).item()
        else:
            activity_2 = sample_deviating_actions(
                1, exclude=[activity, activity_right_before], rng=self.rng
            ).item()
            label = "B"
            answer = sample_binary_no_answer_templates(1, rng=self.rng).item()

        question = (
            self.rng.choice(self.templates_happened_right_before_binary)
            .item()
            .format(activity_1=activity, activity_2=activity_2)
        )

        options_dict = create_options_dict("binary")
        qna_pair = (question, answer, "right_before_binary", "binary", options_dict, label)
        return qna_pair

    def right_before_open(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generate a question-answer pair based on the comparison of the activity before the given action. The returned answer contains the action proceeding the chosen action

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(1, len(actions)).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        question = (
            self.rng.choice(self.templates_happened_right_before_open)
            .item()
            .format(activity=actions[index_anchor])
        )
        answer = (
            sample_open_answer_templates(1, rng=self.rng).item().format(activity=actions[index_anchor - 1])
        )

        qna_pair = (question, answer, "right_before_open", "open", None, actions[index_anchor - 1])
        return qna_pair

    def right_before_multi(
        self, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
    ) -> tuple[str, str, str, str, dict, str]:
        """
        Generates a question and answer pair related to the activity before a chosen activity. Multi choice answer is provided if the activity after the chosen activity is the proceeding one or not.

        Args:
            actions: A sequence of actions.
            time_stamps_end: An array of timestamps indicating the end time of each action.

        Returns:
            A tuple containing textual question, textual answer, question category, answer category, answer options and correct answer.
        """
        if actions.size < 2:
            raise InvalidPromptSequenceError()

        index_anchor = self.rng.integers(1, actions.size).item()
        if not is_unique(index_anchor, actions):
            raise InvalidPromptSequenceError()

        chosen_activity = actions[index_anchor]
        right_before_activity = actions[index_anchor - 1]

        false_activities = sample_deviating_actions(
            2, exclude=[chosen_activity, right_before_activity], rng=self.rng
        ).tolist()

        choices = [right_before_activity, *false_activities]
        assert len(set(choices)) == len(choices)
        indices = self.rng.permutation(3)

        label = determine_multi_letter(indices)
        options_dict = create_options_dict("multi", choices, indices)

        question = (
            self.rng.choice(self.templates_happened_right_before_multi)
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
            .format(solution=right_before_activity)
        )
        qna_pair = (question, answer, "right_before_multi", "multi", options_dict, label)
        return qna_pair
