import random
from collections.abc import Sequence
from math import floor
from pathlib import Path

import numpy as np
from numpy.random import Generator


# read inputs function
def read_inputs(path: str) -> Sequence[str]:
    """
    Read and process an input file to extract non-comment and non-empty lines.

    Args:
        path: The path to the input file.

    Returns:
        A sequence of non-comment and non-empty lines.

    """
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    # read txt file without comments and empty lines
    non_empty_ones = filter(None, lines)
    without_comments = filter(lambda line: not line.startswith("#"), non_empty_ones)
    return np.asarray(list(without_comments))


def read_actions(path) -> np.ndarray:
    """
    Read and process an input file to extract actions.

    Args:
        path (str): The path to the input file.

    Returns:
        Array of actions extracted from the input file; output: [(jumping, hopping), (running, sprinting, walking fast), (clapping)]

    """
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    non_empty_ones = filter(None, lines)
    without_comments = filter(lambda line: not line.startswith("#"), non_empty_ones)
    cleaned_lines = list(without_comments)

    output = [line.split(",") for line in cleaned_lines]

    return output


template_dir = (Path(__file__).parents[2] / "prompts").resolve()
# read actions from txt file
all_actions = read_actions(template_dir / "base" / "list_of_actions_upper_body.txt")
# create generic answer templates
prompt_templates = read_inputs(template_dir / "prompt_templates" / "generic.txt")
prompt_templates_first = read_inputs(template_dir / "prompt_templates" / "first.txt")
prompt_templates_last = read_inputs(template_dir / "prompt_templates" / "last.txt")
open_answer_templates = read_inputs(template_dir / "answer_templates" / "open.txt")
multi_choice_answer_templates = read_inputs(template_dir / "answer_templates" / "multi_choice.txt")
binary_yes_answer_templates = read_inputs(template_dir / "answer_templates" / "binary_yes.txt")
binary_no_answer_templates = read_inputs(template_dir / "answer_templates" / "binary_no.txt")
del template_dir


# create sample templates for prompts and answers
def sample_prompt_templates(number: int, rng: Generator, distinct: bool = False) -> np.ndarray[str]:
    """
    Sample prompt templates from the available options.

    Args:
        number: The number of prompt templates to sample.
        rng: The random number generator to use.
        distinct: Whether to sample distinct templates or allow duplicates.
            Defaults to False.

    Returns:
        An array of sampled prompt templates.
    """
    if number < 0:
        raise InvalidPromptSequenceError
    return rng.choice(prompt_templates, size=number, replace=not distinct, shuffle=True)


def sample_prompt_templates_first(number: int, rng: Generator, distinct: bool = False) -> np.ndarray[str]:
    """see above"""
    return rng.choice(prompt_templates_first, size=number, replace=not distinct, shuffle=True)


def sample_prompt_templates_last(number: int, rng: Generator, distinct: bool = False) -> np.ndarray[str]:
    return rng.choice(prompt_templates_last, size=number, replace=not distinct, shuffle=True)


def sample_open_answer_templates(number: int, rng: Generator, distinct: bool = False) -> np.ndarray[str]:
    return rng.choice(open_answer_templates, size=number, replace=not distinct, shuffle=True)


def sample_multi_choice_answer_templates(
    number: int, rng: Generator, distinct: bool = False
) -> np.ndarray[str]:
    return rng.choice(multi_choice_answer_templates, size=number, replace=not distinct, shuffle=True)


def sample_binary_yes_answer_templates(
    number: int, rng: Generator, distinct: bool = False
) -> np.ndarray[str]:
    return rng.choice(binary_yes_answer_templates, size=number, replace=not distinct, shuffle=True)


def sample_binary_no_answer_templates(number: int, rng: Generator, distinct: bool = False) -> np.ndarray[str]:
    return rng.choice(binary_no_answer_templates, size=number, replace=not distinct, shuffle=True)


def sample_actions(
    number: int,
    similarity_min: int,
    similarity_max: int,
    rng: Generator,
    distinct: bool = False,
    more_similar_actions: bool = False,
) -> np.ndarray[str]:
    """
    Generate a sample of actions based on the provided options.

    Args:
        number: The number of actions to sample.
        similarity_min:
        similarity_max:
        rng: The random number generator to use.
        distinct: Whether the sampled actions should be distinct. Defaults to False.
        more_similar_actions: Whether to force more duplicates in the sampling. Defaults to True.

    Returns:
        An array of sampled actions.

    """
    flat_actions = np.concatenate(all_actions)

    # Create more_similar_actions if bool set on True to force more duplicates
    if more_similar_actions is True:
        # the smaller this range is, the likelier are multiple occurrences of the same action in a prompt sequence
        pre_sample_modificator = rng.integers(similarity_min, similarity_max, endpoint=True).item()
        flat_actions = rng.choice(
            flat_actions,
            size=number * pre_sample_modificator,
            replace=distinct,
            shuffle=True,
        )

    return rng.choice(flat_actions, size=number, replace=not distinct, shuffle=True)


def sample_deviating_actions(number: int, exclude: np.ndarray[str], rng: Generator) -> np.ndarray[str]:
    """
    Sample a specified number of deviating actions from a given array of merged actions.
    Deviating actions are sampled until a set of actions is obtained that does not contain any synonyms
    with the merged actions.

    Prevents duplicate actions in the returned actions.

    Args:
        number: The number of deviating actions to sample.
        excluded: An array of actions to not sample.
        rng: The random number generator to use.

    Returns:
        An array of sampled deviating actions.

    Raises:
        ValueError: If the function fails to sample deviating actions after 100 tries.
    """

    iteration_counter = 0
    while True:
        if iteration_counter > 100:
            raise InvalidPromptSequenceError(
                'Function "sample_deviating_actions" was not able to sample actions in 100 tries.'
            )
        iteration_counter += 1

        # The actual sampoling attempt
        synonym_found = False
        sample = sample_actions(number, 1, 1, rng, distinct=True, more_similar_actions=False)
        for move in sample:
            for prompt in exclude:
                if is_synonymous(move, prompt):
                    synonym_found = True
        if not synonym_found:
            return sample


def sample_timestamps(
    num_steps: int,
    entire_time_length: float,
    min_fraction_of_time_per_step: float,
    rng: Generator,
) -> np.ndarray[np.float64]:
    """
    Sample timestamps for a given number of steps within a specified time range. The difference between consecutive timestamps is sampled uniformly.
    The timestamps always start with zero and end with a value strictly less than `entire_time_length`.

    Args:
        num_steps: The number of timestamps to sample.
        entire_time_length: The total time length within which the timestamps should be sampled.
        min_fraction_of_time_per_step (float): The minimum fraction of time that each step should receive,
            relative to a single step's time when making each step as long as all others.
            For example, if `num_steps` is 5 and `entire_time_length` is 10 seconds,
            each step would initially have `entire_time_length / 5 == 2.0` seconds.
            If `min_fraction_of_time_per_step == 0.25`, then each step would have at least
            `2.0 * min_fraction_of_time_per_step == 0.5` seconds of length.
            If set to 1.0, each step would have the exact same length of `2.0` seconds.
        rng: The random number generator to use.

    Returns:
        An array of sampled timestamps.

    Raises:
        AssertionError: If `num_steps` is not greater than 1.
    """
    assert num_steps > 1

    # Make random numbers that sum up to 1
    random_part = rng.random(num_steps)  # this is uniform sampling
    random_part /= random_part.sum()

    # Make a vector of all the minimum time lengths
    assert min_fraction_of_time_per_step > 0.0
    mins = np.full((num_steps,), (1 / num_steps) * min_fraction_of_time_per_step, dtype=np.float64)

    # Combine them
    normalized_time_steps = mins + (1 - min_fraction_of_time_per_step) * random_part
    timestamps = (normalized_time_steps * entire_time_length).cumsum()

    # We want to start with zero instead of ending with entire_time_length
    return timestamps - timestamps[0]


def is_synonymous(string1: str, string2: str) -> bool:
    """
    Check if two strings are synonyms based on a collection of subarrays.

    Args:
        string1: The first string to compare.
        string2: The second string to compare.

    Returns:
        True if the strings are synonyms, False otherwise.
    """
    for subarray in all_actions:
        if string1 in subarray and string2 in subarray:
            return True
    return False


def what_is_happening_at_t(
    time: np.float64, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
) -> str:
    """
    Determine the activity that is happening at a given time.

    Args:
        time: The time for which to determine the activity.
        actions: An array of actions.
        time_stamps_end: An array of end timestamps corresponding to each action.

    Returns:
        The activity that is happening at the given time.
    """
    return actions[np.searchsorted(time_stamps_end, time)]


def is_unique(index: int, input_array: np.ndarray[str]) -> bool:
    """
    Check if the element at the specified index in the input array is unique.

    Args:
        index: The index of the element to check.
        input_array: The input array to compare the element against.

    Returns:
        True if the element at the specified index is unique, False otherwise.
    """
    element = input_array[index]
    for i, item in enumerate(list(input_array)):
        if i != index and is_synonymous(element, item):
            return False
    return True


def count_occurrences(action: str, all_actions: np.ndarray[str]) -> int:
    """
    Count the number of occurrences of a specific action in an input array.

    Args:
        action: The action to count occurrences for.
        input_array: The input array to search for occurrences.

    Returns:
        The count of occurrences of the specified action in the input array.
    """
    counter = 0
    for element in all_actions:
        if is_synonymous(element, action):
            counter += 1
    return counter


def count_actions_in_interval(
    start: float, end: float, actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
) -> int:
    """
    Count the number of actions that fall within a specified time interval.

    If an action starts before the start time of the interval but ends after the start time of the interval,
    it is counted as well.

    Args:
        start: The start time of the interval.
        end: The end time of the interval.
        actions: Array of actions.
        time_stamps_end: Array of end timestamps for each action. Assumed to be sorted in ascending order.

    Returns:
        The count of actions that fall within the specified time interval.
    """
    index_start = np.searchsorted(time_stamps_end, start)
    index_end = np.searchsorted(time_stamps_end, end)
    return len(set(actions[index_start : index_end + 1]))


def determine_multi_letter(indices: list[int]) -> str:
    """
    Find out which option out of A, B and C is the correct one based on the index of the correct answer which is 0.

    Args:
        indices: indices of the answer options

    Returns:
        the correct answer option
    """
    assert len(indices) == 3

    if indices[0] == 0:
        return "A"
    elif indices[1] == 0:
        return "B"
    else:
        return "C"


def create_options_dict(
    mode: str, options: list[str] | None = None, indices: list[int] | None = None
) -> dict | None:
    """
    Creates the options-dict of each instance of qa_pairs depending on the answer type

    Args:
        mode: Is either open, multi or binary
        options: only relevant for multi questions. represents the answer options
        indices: indices of the answer options

    Returns:
        a dict containing the answer options
    """
    if mode == "binary":
        return {"A": True, "B": False}
    elif mode == "open":
        return None
    else:
        result = {}
        for letter, index in zip(["A", "B", "C"], indices):
            assert index < len(options)
            result[letter] = options[index]
        return result


def add_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Fuses 2 dicts with equal keys

    Args:
        dict1: self explanatory
        dict2: self explanatory

    Returns:
        a dict containing the data of both input dicts
    """
    result = {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}
    return result


def filter_dict_by_keys(mapping: dict, filter: list) -> dict:
    """
    Deletes all keys in a dict that are not contained in list

    Args:
        mapping: self explanatory
        filter: list of keys that should be kept in dict

    Returns:
        the filtered dictionary
    """
    return {key: value for key, value in mapping.items() if key in filter}


def merge_actions(
    actions: np.ndarray[str], time_stamps_end: np.ndarray[np.float64]
) -> tuple[np.ndarray[str], np.ndarray[np.float64]]:
    """
    Merge consecutive actions that have the same meaning (or are the same) into a single action.

    Args:
        actions: Array of actions.
        time_stamps_end: Array of end timestamps for each action.

    Returns:
        Tuple containing the merged actions and their corresponding end timestamps.
    """

    merged_actions = []
    merged_time_stamps_end = []
    for i in range(actions.size):
        if not merged_actions or not is_synonymous(actions[i], merged_actions[-1]):
            merged_actions.append(actions[i])
            merged_time_stamps_end.append(time_stamps_end[i])
        else:
            # this means that the current action is the same as the last one
            merged_time_stamps_end[-1] = time_stamps_end[i]  # update the end timestamp of the last action

    return np.asarray(merged_actions), np.asarray(merged_time_stamps_end)


def format_list_to_string(strings: list) -> str:
    """
    Formats a list of strings into a single string with proper punctuation.

    This is used to concatenate arrays as in questions and answer templates to look more pretty.

    Args:
        strings: The list of strings to be formatted.

    Returns:
        The formatted string.
    """
    length = len(strings)
    if length == 0:
        return ""
    if length == 1:
        return strings[0]
    elif length == 2:
        return strings[0] + " and " + strings[1]
    else:
        return ", ".join(strings[:-1]) + ", and " + strings[-1]


def text_description(prompt_sequence: list) -> str:
    """
    Generates a textual (more fancy) description based on the given prompt sequence.

    Args:
        prompt_sequence:
            A list of tuples representing the prompt sequence.
            Each tuple contains the start time, end time, and prompt for a specific action.

    Returns:
        The generated textual description.
    """

    def to_str(number):
        return f"%.{1}f" % round(number, 1)

    def to_timestamp(number):
        """Display as minutes:seconds"""
        return f"{int(floor(number / 60))}:{to_str(number % 60)}"

    return " ".join(
        # Choose a random template to generate diverse texts
        random.choice(
            [
                f"{prompt} from {to_timestamp(start)} to {to_timestamp(end)} (for {to_str(end-start)} seconds).",
                f"{prompt}. This is happening from {to_timestamp(start)} to {to_timestamp(end)}, which means for a total time of {to_str(end-start)} seconds.",
                f"{prompt} from {to_timestamp(start)} to {to_timestamp(end)} ({to_str(end-start)} seconds total).",
                f"{prompt}. This action is starting at {to_timestamp(start)} and continues until {to_timestamp(end)}, resulting in a total time of {to_str(end-start)} seconds.",
                f"{prompt}. This action is starting at {to_timestamp(start)} and happens until {to_timestamp(end)}, meaning the engagement in this is for {to_str(end-start)} seconds.",
                f"{prompt}. It is starting at second {to_timestamp(start)} and continuing until {to_timestamp(end)}, so the person is doing this for {to_str(end-start)} seconds.",
                f"{prompt} at timestamp {to_timestamp(start)} and ending at {to_timestamp(end)}, resulting in a total time of {to_str(end-start)} seconds.",
            ]
        )
        for start, end, prompt in prompt_sequence
    )


class InvalidPromptSequenceError(Exception):
    """
    An error raised when the prompt sequence of the current iteration does not fit the randomly chosen question type.
    """
