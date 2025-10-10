from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

from ..qna_type import (
    ActionCountQuestion,
    AfterQuestion,
    BeforeQuestion,
    ComparisonAction,
    ComparisonCounting,
    DescriptiveIdentificationQuestion,
    ExtremumQuestion,
    FirstQuestion,
    IntervalQuestion,
    LastQuestion,
)
from .data_instance import DataInstance
from .utility import (
    merge_actions,
    sample_actions,
    sample_prompt_templates,
    sample_prompt_templates_first,
    sample_prompt_templates_last,
    sample_timestamps,
    text_description,
)


def sample_data_instance(
    index: int,
    min_actions: int,
    max_actions: int,
    entire_time_length: float,
    rng: Generator,
    num_qa_pairs_each: int = 3,
    min_fraction_of_time_per_step: float = 0.3,
    similarity_factor_min: int = 1,
    similarity_factor_max: int = 3,
    answer_types: str = "all",
    question_type_distribution: Sequence[float] = [0.1] * 10,
    distinct_actions: bool = False,
    distinct_templates: bool = False,
) -> DataInstance:
    """
    Generates a single instance of a data sample.

    Args:
        index: Index of the data instance.
        min_actions: Minimum number of actions.
        max_actions: Maximum number of actions.
        entire_time_length: Length of the entire time interval.
        rng: Random number generator to fix the seed for reproducibility.
        num_qa_pairs_each: Number of question-answer pairs for each data instance.
        min_fraction_of_time_per_step: Minimum fraction of time per step.
        similarity_factor_min: Minimum similarity factor.
        similarity_factor_max: Maximum similarity factor.
        answer_types: The types of answers to generate.
        question_type_distribution: A list of probabilities for each question type. Should be sum to 1.
        distinct_actions: Whether to use distinct actions.
        distinct_templates: Whether to use distinct templates.

    Returns:
        A data instance object.

    """

    # sample parameter and templates
    num_actions = rng.integers(min_actions, max_actions, endpoint=True).item()
    actions = sample_actions(
        num_actions,
        similarity_factor_min,
        similarity_factor_max,
        rng=rng,
        distinct=distinct_actions,
        more_similar_actions=True,
    )
    prompt_templates_middle = sample_prompt_templates(num_actions - 2, rng=rng, distinct=distinct_templates)
    prompt_templates_first = sample_prompt_templates_first(1, rng=rng, distinct=distinct_templates)
    prompt_templates_last = sample_prompt_templates_last(1, rng=rng, distinct=distinct_templates)

    # join templates
    templates_concatenated = np.concatenate(
        (prompt_templates_first, prompt_templates_middle, prompt_templates_last)
    )

    # fill joined templates with actions
    text_prompts_advanced = (
        template.format(activity=action, Activity=(action[0].capitalize() + action[1:]))
        for template, action in zip(templates_concatenated, actions)
    )

    time_stamps_start = sample_timestamps(
        num_actions,
        entire_time_length=entire_time_length,
        min_fraction_of_time_per_step=min_fraction_of_time_per_step,
        rng=rng,
    )
    time_stamps_end = np.concatenate((time_stamps_start[1:], (entire_time_length,)))

    # create list of prompt_sequence with timestamps and text_prompts as well as the textual description
    prompt_sequence = list(zip(time_stamps_start, time_stamps_end, text_prompts_advanced))
    textual_description = text_description(prompt_sequence)

    # Generate question-answer-pair
    qna_actions, qna_time_stamps_end = merge_actions(actions, time_stamps_end)
    qa_pairs = [
        sample_qna_tuple(
            qna_actions,
            qna_time_stamps_end,
            question_type_distribution=question_type_distribution,
            answer_types=answer_types,
            rng=rng,
        )
        for _ in range(num_qa_pairs_each)
    ]

    # Check integrity of the Q&A pairs
    for _, answer, question_type, answer_type, answer_options, correct_answer in qa_pairs:
        # Check for some really bad errors
        match answer_type:
            case "multi":
                assert correct_answer in answer_options
                assert len(set(answer_options.values())) == len(
                    answer_options.values()
                ), f"Duplicate answers {answer_options} in {question_type}"
            case "open":
                assert correct_answer in answer  # subsequence must be present in the answer

    # keep stats of every iteration
    stats = {
        "InvalidPromptSequenceErrors": 0,
        "question_types": defaultdict(int),
        "answer_types": defaultdict(int),
        "correct_answers": defaultdict(int),
    }
    for _, _, question_type, answer_type, _, correct_answer in qa_pairs:
        stats["question_types"][question_type] += 1
        stats["answer_types"][answer_type] += 1
        stats["correct_answers"][correct_answer] += 1

    return DataInstance(
        index=index,
        actions=actions,
        stats=stats,
        prompt_sequence=prompt_sequence,
        textual_description=textual_description,
        qa_pairs=qa_pairs,
    )


def sample_qna_tuple(
    actions: Sequence[str],
    time_stamps_end: np.ndarray,
    question_type_distribution: Sequence[float],
    answer_types: str,
    rng: Generator,
) -> tuple[str, str, str, str, dict, str]:
    """
    Samples a question and its corresponding answer pair, along with additional information (time_stamp as well as the assigned label).

    Args:
        actions: A sequence of action strings.
        time_stamps_end: An array of end timestamps corresponding to each action.
        question_type_distribution: A numpy array of probabilities for each question type. Does not need to sum to 1.
        answer_types: The types of answers to generate.
        rng: Random number generator to use.

    Returns:
        A tuple containing the textual question, textual answer, question category, answer category,
        answer options and correct answer:
        ``(question, answer, question_type, answer_type, answer_options, correct_answer)``.

    """
    questions = [
        AfterQuestion(rng=rng),
        BeforeQuestion(rng=rng),
        LastQuestion(rng=rng),
        FirstQuestion(rng=rng),
        ComparisonAction(rng=rng),
        ComparisonCounting(rng=rng),
        ActionCountQuestion(rng=rng),
        ExtremumQuestion(rng=rng),
        IntervalQuestion(rng=rng),
        DescriptiveIdentificationQuestion(rng=rng),
    ]
    if not len(questions) == len(question_type_distribution):
        raise ValueError("The number of question types and the number of probabilities must be the same.")

    question_type_distribution = np.asarray(question_type_distribution, dtype=np.float32)
    question_type_distribution /= question_type_distribution.sum()

    question = rng.choice(a=questions, p=question_type_distribution)

    return question.sample_question_answer_pair(actions, time_stamps_end, answer_types)
