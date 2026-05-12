
import os
import sys
import traceback

sys.path.append(os.getcwd())

from generator.algo_human import solve_using_human_techniques, TECHNIQUES, TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT
from generator.verification import solve, check_is_unique_solution, NoSolutionException

from tool_logs.parser import process_args, SyntaxInvalidException, InputsInvalidException
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
