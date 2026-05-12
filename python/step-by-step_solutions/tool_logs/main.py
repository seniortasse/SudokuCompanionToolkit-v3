
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

# Make legacy sibling packages importable regardless of the current working directory.
#
# This script lives at:
#   python/step-by-step_solutions/tool_logs/main.py
#
# The sibling packages live under:
#   python/step-by-step_solutions/generator
#   python/step-by-step_solutions/tool_logs
#
# Therefore the directory that must be on sys.path is:
#   python/step-by-step_solutions
_LEGACY_ROOT = Path(__file__).resolve().parents[1]
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from generator.algo_human import solve_using_human_techniques, TECHNIQUES, TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT
from generator.model import EMPTY_CHAR, Instance
from generator.verification import solve, check_is_unique_solution, NoSolutionException

from tool_logs.parser import process_args, SyntaxInvalidException, InputsInvalidException
import tool_logs.writer as writer_module
from tool_logs.writer import write_steps


# When adding a new technique, have to define:
#  - message (messages.py)
#    - additionally, add to map_technique_names on top of file
#  - coloring (formatting.py)
#    - additionally, establish whether it is a group 1/2 technique, which use partial information for formatting
#  - identify earlier applications (layers.py)


TOOL_NAME = "tool_logs"

TOOL_MODES = [
    (
        "Single instance",
        [
            ("", "{file_name}", "Input file containing the instance to be solved (.xlsx file)"),
        ]
    ),
    (
        "Multiple instances",
        [
            ("--list", "{file_name}", "Input file containing a list of instances to be solved (.xlsx file)"),
        ]
    ),
]




PUBLISHING_BRIDGE_FLAGS = {
    "--input-json",
    "--output",
    "--locale",
    "--template",
    "--messages",
}


def _is_publishing_bridge_invocation(args):
    """
    Detect invocation from the modern publishing pipeline.

    The legacy tool originally accepted only:
        main.py instance.xlsx
        main.py --list instances.xlsx

    The publishing pipeline calls:
        main.py --input-json ... --output ... --locale ... --template ... --messages ...
    """

    return any(arg in PUBLISHING_BRIDGE_FLAGS for arg in args)


def _parse_publishing_bridge_args(args):
    parser = argparse.ArgumentParser(
        description="Publishing bridge for generating one Sudoku user log workbook."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--messages", required=True)
    return parser.parse_args(args)


def _grid_from_givens81(givens81):
    """
    Convert a flattened givens81 string into the legacy generator Instance grid.

    Legacy empty cells are represented by EMPTY_CHAR, which is a single space.
    """

    value = str(givens81 or "").strip()
    if len(value) != 81:
        raise ValueError(f"Expected givens81 to contain 81 characters, got {len(value)}.")

    grid = []
    for row_index in range(9):
        row = []
        for col_index in range(9):
            char = value[row_index * 9 + col_index]
            if char in ("0", "."):
                row.append(EMPTY_CHAR)
            elif char in "123456789":
                row.append(char)
            else:
                raise ValueError(f"Unexpected givens81 character: {char!r}")
        grid.append(row)

    return grid


def _instance_id_from_output_path(output_path, payload):
    """
    Determine the workbook instance id.

    For output:
        .../L-1-1_user_logs.xlsx

    we want:
        L-1-1

    because writer.write_steps appends "_user_logs.xlsx" itself.
    """

    external_puzzle_code = str(payload.get("external_puzzle_code") or "").strip()
    if external_puzzle_code:
        return external_puzzle_code

    stem = Path(output_path).stem
    if stem.endswith("_user_logs"):
        return stem[: -len("_user_logs")]
    return stem


def _run_solver_for_instance(instance):
    """
    Shared solving logic for the publishing bridge.

    This mirrors the existing legacy main flow:
        1. solve/check uniqueness
        2. solve using human techniques
        3. return logs_steps
    """

    try:
        solution = solve(instance)
    except NoSolutionException:
        raise Exception("No solution for instance!")

    has_unique_solution = check_is_unique_solution(instance, solution)
    assert has_unique_solution, "No unique solution for instance!"

    print("Solving using human techniques..")

    techniques_disabled = (
        TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT
        if instance.uses_custom_boxes_layout
        else []
    )
    if techniques_disabled:
        print(f"Disabled techniques {techniques_disabled} as the solution uses a custom boxes layout")

    use_techniques = [
        technique
        for technique in TECHNIQUES
        if technique not in techniques_disabled
    ]

    solved_instance, (_, logs) = solve_using_human_techniques(
        instance,
        use_techniques=use_techniques,
        magic_solution=solution,
        show_logs=False,
        max_number_iterations=None,
    )

    assert solved_instance == solution

    return logs["steps"]


def _run_publishing_bridge(args):
    """
    Generate one localized Excel user_logs workbook from publishing JSON payload.

    Input JSON is produced by:
        python/publishing/step_solutions/log_generator.py

    This bridge deliberately does not import publishing modules. The publishing
    side owns all modern paths, names, locale selection, and package structure.
    The legacy side only solves and writes the workbook.
    """

    parsed = _parse_publishing_bridge_args(args)

    input_json_path = Path(parsed.input_json)
    output_path = Path(parsed.output)
    visual_template_path = Path(parsed.template)
    message_template_path = Path(parsed.messages)

    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    if not visual_template_path.exists():
        raise FileNotFoundError(f"Visual template not found: {visual_template_path}")
    if not message_template_path.exists():
        raise FileNotFoundError(f"Message template not found: {message_template_path}")

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))

    givens81 = str(payload.get("givens81") or "").strip()
    if not givens81:
        raise ValueError(f"Missing givens81 in input JSON: {input_json_path}")

    instance_id = _instance_id_from_output_path(output_path, payload)
    instance = Instance(_grid_from_givens81(givens81))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make the legacy writer use the canonical publishing templates selected
    # by the modern workflow.
    writer_module.FILE_NAME_TEMPLATE = str(visual_template_path)
    writer_module.FILE_NAME_TEMPLATE_MESSAGES = str(message_template_path)

    logs_steps = _run_solver_for_instance(instance)

    print()
    print("Generating output..")
    print("Locale:", parsed.locale)
    print("Input JSON:", input_json_path)
    print("Output workbook:", output_path)
    print("Visual template:", visual_template_path)
    print("Message template:", message_template_path)

    data_folder = str(output_path.parent)
    if data_folder and not data_folder.endswith(("/", "\\")):
        data_folder += os.sep

    file_name_output = writer_module.write_steps(
        instance,
        logs_steps,
        instance_id,
        data_folder,
    )

    generated_path = Path(file_name_output)
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Legacy writer reported output {generated_path}, but the file does not exist."
        )

    if generated_path.resolve() != output_path.resolve():
        # This should normally not happen, but if it does, copy into the exact
        # publishing path expected by the package workflow.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated_path.read_bytes())

    print("Publishing bridge completed:", output_path)



if __name__ == "__main__":
    """
    Steps:
      1 Process command line arguments
      2 Generate logs by applying algorithm solving the instance using human techniques
      3 Read formatting from template
      4 Write steps to output file
    """

    def format_bold(text):
        return "\033[1m" + text + "\033[0m"

    syntax = '\n'.join(
        [
            '\n\n'.join([
                '\n'.join([
                    format_bold(tool_mode),
                    "  python {}/main.py {}".format(
                        TOOL_NAME,
                        " ".join(
                            # Only add whitespace if argument is named
                            arg[0] + (" " if arg[0] else "") + arg[1]
                            for arg in tool_args
                        )
                    ),
                    # Explanation of arguments
                    *(
                        "  {}: {}".format(
                            arg[1], arg[2]
                        )
                        for arg in tool_args
                    )
                ])
                for tool_mode, tool_args in TOOL_MODES
            ]),
            "",
            # Comments
            *(
                "The tool expects a \"Template.xlsx\" file specifying the format to be used",
            )
        ]
    )

    # 1
    args = sys.argv[1:]

    # Publishing bridge mode.
    #
    # This must run before the old parser because the old parser only accepts:
    #   main.py input.xlsx
    #   main.py --list instances.xlsx
    #
    # The modern publishing pipeline passes:
    #   --input-json --output --locale --template --messages
    if _is_publishing_bridge_invocation(args):
        try:
            _run_publishing_bridge(args)
        except Exception as e:
            print()
            print("[ERROR] Publishing bridge failed.")
            print()
            print("Message:", str(e))
            print()
            print(traceback.format_exc(limit=20))
            raise
        raise SystemExit(0)

    # Check for help command
    if len(args) == 1 and args[0] in ["-h", "--help"]:
        print("The following syntax is used:")
        print()
        print(syntax)
        quit()

    # Process arguments
    try:
        data_folder, instances = process_args(args)
    except (SyntaxInvalidException, InputsInvalidException) as e:
        print("Syntax/Inputs invalid!")
        print()
        print("Error message:", str(e))
        if isinstance(e, SyntaxInvalidException):
            print()
            print("Please provide input using the following syntax:\n\n", syntax)
        quit()

    print()

    # User confirmation
    num_instances = len(instances)
    if num_instances > 1:
        while True:
            _input = input(f"Trying to run for {num_instances} instances from file, continue? [y/n] ")
            print("Input:", _input)
            if _input == "y":
                print("Continuing..")
                break
            elif _input == "n":
                print("Aborting..")
                quit()
            else:
                print("Input not recognised, please try again..")
                continue
        print()

    for instance_id, instance in instances:

        # Pre-check: There is a unique solution
        try:
            solution = solve(instance)
        except NoSolutionException:
            raise Exception("No solution for instance!")

        has_unique_solution = check_is_unique_solution(instance, solution)
        assert has_unique_solution, "No unique solution for instance!"

        # 2
        print("Solving using human techniques..")

        # When generating an instance for a solution with a non-standard boxes layout, do not include the boxes-x
        #  techniques, as their implementation requires a standard boxes layout
        # TODO It would be much better to do this internally, as this logic has now been scattered through several
        #  places in the codebase
        techniques_disabled = TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT if instance.uses_custom_boxes_layout else []
        if techniques_disabled:
            print(f"Disabled techniques {techniques_disabled} as the solution uses a custom boxes layout")
        use_techniques = [technique for technique in TECHNIQUES if technique not in techniques_disabled]

        solved_instance, (_, logs) = solve_using_human_techniques(
            instance, use_techniques=use_techniques, magic_solution=solution,
            show_logs=False, max_number_iterations=None
        )
        logs_steps = logs["steps"]

        assert solved_instance == solution

        # 3 & 4
        print()
        print("Generating output..")

        try:
            file_name_output = write_steps(instance, logs_steps, instance_id, data_folder)
        except Exception as e:
            print()
            print(f"[ERROR] Something went wrong when trying to generate user logs for instance \"{instance_id}\"")
            print()
            print("Message:", str(e))
            print()
            print(traceback.format_exc(limit=10))
            print()
