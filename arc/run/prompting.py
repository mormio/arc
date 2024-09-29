# adapted from https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/master/arc_solve/prompting.py

from typing import List

import numpy as np

from arc.data import grid_to_ascii

ASCII_GRID_SIZE_LIM = 35

color_scheme_consts_name = {
    0: "BLACK",
    1: "BLUE",
    2: "PURPLE",
    3: "GREEN",
    4: "YELLOW",
    5: "GREY",
    6: "FUSCHIA",
    7: "ORGANE",
    8: "TEAL",
    9: "BROWN",
}


def make_user_prompt(
    examples: List[dict],
    model_name: str,
    primitives_source_codes: List[str] = None,
) -> str:
    prompt = "In this problem, the example pairs of grid inputs and outputs are the following:"
    prompt += format_ascii_examples(examples)
    prompt += "\n\n"
    prompt += "And, te provided primitives are the following: \n\n"
    prompt += format_primitives_source_codes(primitives_source_codes)
    prompt += "Your answer:"

    if model_name in ["gpt-4o", "gpt-4o-gs"]:
        return [{"type": "text", "content": prompt}]
    else:
        return prompt


def format_primitives_source_codes(
    primitives_source_codes: List[str] = None,
) -> str:
    if not primitives_source_codes:
        primitives_text = "[NONE AVAILABLE]"
        return primitives_text
    return "".join(primitives_source_codes)


def format_ascii_examples(examples: List[dict]) -> str:
    ascii_text = ""

    for ex in examples:
        input = ex["input"]
        output = ex["output"]
        for grid_name, grid in zip(("###\nInput", "Output"), (input, output)):
            try:
                array = np.array(grid)
                x, y = array.shape

                if max(y, x) > ASCII_GRID_SIZE_LIM:
                    ascii_grid = "[OMITTED DUE TO SIZE CONSTRAINTS]"
                ascii_grid = grid_to_ascii(grid=array, separator="|")

                shape_text = f" (shape: {x} by {y})"

            except (SyntaxError, ValueError) as e:
                print(
                    "Invalid input string. Must be a valid representation of a list of lists.",
                    e,
                )
                shape_text = " (shape: [OMITTED DUE TO FORMATTING ERROR])"
                ascii_grid = "[OMITTED DUE TO FORMATTING ERROR]"

            ascii_text += f"\n\n{grid_name}{shape_text}:\n\n{ascii_grid}"

    return ascii_text


def make_system_prompt(recommend_primitives=True) -> str:
    input_line = make_input_line()
    maybe_diff_highlight_line = ""
    maybe_diff_triangles_line = ""
    recommend_primitives_line = (
        make_recommend_primitives_line() if recommend_primitives else ""
    )
    additional_info_line = make_additional_info_line(long_as_you_want=False)
    prompt = _format_prompt(
        input_line,
        maybe_diff_highlight_line,
        maybe_diff_triangles_line,
        additional_info_line,
        recommend_primitives_line,
    )
    return prompt


def make_additional_info_line(long_as_you_want=False) -> str:
    additional_info_line_reasoning = """You follow a particular reasoning style.
    You break down complex problems into smaller parts and reason through them step
    by step, arriving at sub-conclusions before stating an overall conclusion. This
    reduces the extent to which you need to do large leaps of reasoning."""
    no_need_conside_as_long = """\n\nYour reasoning **can be as long as necessary**!
    The goal of the reasoning is just to make sure you end up with a correct
    implementation of the transformation rule, so **there isn't any need
    for your reasoning to be concise**. You should do any and all reasoning that
    would be useful."""

    if not long_as_you_want:
        no_need_conside_as_long = ""
    additional_info_line_attributes = (
        "You are creative and accomplished at solving puzzles."
    )
    additional_info = f"{additional_info_line_reasoning}{no_need_conside_as_long}\n\n{additional_info_line_attributes}"
    return additional_info


def make_input_line() -> str:
    scheme = color_scheme_consts_name
    color_to_index = ", ".join(
        f"{color_val}: {name.capitalize()}"
        for color_val, name in enumerate((scheme).values())
    )
    many_ascii_rep_and_skip_image_version_of_input_line = f"""The inputs and outputs are each "grids".
    A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown
    to you in various ASCII representations. Each number corresponds to a color. The correspondence
    is as follows: {color_to_index}. """

    return many_ascii_rep_and_skip_image_version_of_input_line


def make_recommend_primitives_line() -> str:
    line = """You will also be given a list of python primitives that you are encouraged to
    use in your answer. You can also write new functions, but they must be nested within
    your `transform` function. Assume all these primitives and their types will be imported; don't
    import or redefine them in your final answer."""
    return line


def _format_prompt(
    input_line: str,
    recommend_primitives_line: str,
    maybe_diff_highlight_line: str = "",
    maybe_diff_triangles_line: str = "",
    additional_info_line: str = "",
) -> str:
    """This is the prompt that (I think) was used in
    https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt
    """
    alternative_system_prompt_text = f"""You will be given some number of paired example inputs and outputs.
    The outputs were produced by applying a transformation rule to the inputs. In addition to the paired
    example inputs and outputs, there is also one additional input without a known output. Your task is to
    determine the transformation rule and implement it in code.

    {input_line}{maybe_diff_highlight_line}{maybe_diff_triangles_line}{recommend_primitives_line}

    The transformation only needs to be unambiguous and applicable to the example inputs and the additional
    input. It doesn't need to work for all possible inputs.

    You'll need to carefully reason in order to determine the transformation rule. Start your response by
    carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

    After your reasoning write code in triple backticks (```python and then ```). You should write a function
    called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the
    transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the
    transformation which works in general (it shouldn't just work for the additional input).

    Don't write tests in your python code, just output the `transform` function. (It will be tested later.)
    {additional_info_line}"""
    return alternative_system_prompt_text
