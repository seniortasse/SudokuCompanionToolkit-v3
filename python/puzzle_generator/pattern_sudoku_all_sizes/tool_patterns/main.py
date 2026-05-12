
import os
import random
import sys
import time
import traceback

sys.path.append(os.getcwd())

from generator.algo_human import TECHNIQUES

from tool_patterns.parser import process_inputs, SyntaxInvalidException, InputsInvalidException, VALID_SIZES, VALID_RUN_MODES, SIZES_REQUIRING_LAYOUT
from tool_patterns.algorithm_2 import run_algorithm
from tool_patterns.logs import generate_logs
from tool_patterns.output import write_to_excel_file, write_list_of_patterns_to_excel_file
from tool_patterns.logs_algorithm import write_logs_to_file


# TODO Next version: Remove fitter code from the tool altogether, as it is not used in algorithm_2 anymore


# PARAMETERS


# DATA SETTINGS


def main():
    """
    Steps:
      1 Read, process & validate input arguments
      2 Algorithm repetition loop: 
        Version 1: (1) Fit subwords, (2) Apply pattern, (3) Determine whether the resulting instance is acceptable
        Version 2: (1) Generate an instance by filling in values on the black squares of the pattern one-by-one, using
         constraint propagation and a backtracking mechanism, (2) & (3) same
      3 Determine whether the resulting instance is solvable using implemented human techniques (moved to algorithm)
      4 Write output to file
      5 Generate logs with step-by-step overview of techniques used to solve the instance (only for run mode --single)
      6 Generate logs with step-by-step overview of the instance generation algorithm for patterns (only for run mode
        --single)
    """

    # Constants
    # These are not anymore, as the same main file is used for different grid sizes and now given in the input arguments

    tool_name = f"tool_patterns"
    syntax = "\n ".join([
        "  python {}/main.py".format(tool_name) + " {size}-{layout}" + " \"characters\"" + " {run_mode} {file_name_pattern} {options}",
        "",
        "  size:              should be in {}".format(VALID_SIZES),
        "  layout:            optional default layout for grid sizes {} as {{box_width}}x{{box_height}} [only specify when no custom layout is specified]".format(SIZES_REQUIRING_LAYOUT),
        "  run_mode:          {}".format(VALID_RUN_MODES),
        "  file_name_pattern: the file containing a single pattern or list of patterns (.xlsx)",
        "",
        "  options:",
        "   -n value1-value2 (range of number of different techniques used)",
        "   -w value1-value2 (range of weight)",
        "   -t \"technique1, technique2, ...\" (list of techniques required - spaces not necessary, quotes necessary)",
        "      available techniques: " + ", ".join(TECHNIQUES),
        "   -e \"technique1, technique2, ...\" (list of techniques excluded - spaces not necessary, quotes necessary)",
        "      available techniques: " + ", ".join(TECHNIQUES),
        "   -f File name of template for algorithm logs (.xlsx)",
        "   -b File name of file containing custom layout for boxes (.xlsx) [only specify when no default layout is specified]",
    ])

    print()

    # 1 Read, process & validate input arguments
    args = sys.argv[1:]

    # Check for help command
    if len(args) == 1 and args[0] in ["-h", "--help"]:
        print("The following syntax is used:")
        print()
        print(syntax)
        quit()

    # Process input arguments
    try:
        run_mode, file_name_pattern, chars, patterns, options_steering, file_name_template = process_inputs(args)
    except (SyntaxInvalidException, InputsInvalidException) as e:
        print("Syntax/Inputs invalid!")
        print()
        print("Error message:", str(e))
        if isinstance(e, SyntaxInvalidException):
            print()
            message = '\n\n'.join(["Please provide input using the following syntax:", syntax])
            print(message)
        quit()

    # Log input data
    print()
    print("Generating solution(s) and instance(s) for {} pattern(s)".format(len(patterns)))
    print()
    # print("Grid size:   {}".format(size))
    # print("Layout:      {}".format(layout))
    print("Characters:  {}".format(chars))
    print()
    option_names = {
        "-n": "Number of different techniques used",
        "-w": "Weight",
        "-t": "Techniques required",
        "-e": "Techniques excluded",
    }
    print("Steering options:")
    for option_steering in options_steering.items():
        print(" %s: %s" % (option_names[option_steering[0]], option_steering[1]))
    print()

    session_id = int(time.time())

    # To make it easier to reproduce runs, always initialise the seed, which can furthermore by fixed when testing
    seed = random.randint(1, 100_000)
    # seed = 1
    random.seed(seed)
    print()
    print("Initialising tool with random seed:", seed)
    print()

    # Split flow for different run modes
    if run_mode == "--single":
        assert len(patterns) == 1
        _, pattern = patterns[0]
        run_for_single_pattern(chars, pattern, options_steering, session_id, file_name_template)
    elif run_mode == "--list":
        run_for_list_of_patterns(chars, patterns, options_steering, session_id, file_name_pattern)
    else:
        raise NotImplementedError(f"Generating instances with run mode '{run_mode}' not implemented")

    print()
    print("Finished")


def run_for_list_of_patterns(chars, patterns, options_steering, session_id, file_name_pattern):

    # Ask for user confirmation once, to review and confirm the number of patterns and steering criteria
    while True:
        _input = input("Continue? [y/n] ")
        if _input == "y":
            break
        elif _input == "n":
            print("Aborting..")
            quit()
        else:
            print("Input not recognised, try again..")

    
    instances_and_solutions = []
    total_patterns = len(patterns)

    for idx, (pattern_id, pattern) in enumerate(patterns, start=1):

        print()
        print("=" * 72)
        print(f"REQUEST {idx}/{total_patterns}  |  pattern_id={pattern_id}")
        print("=" * 72)
        print("Pattern:")
        for row in pattern:
            print(["o" if e else " " for e in row])
        print()

        # 2 Algorithm
        try:
            solution, instance, logs_algorithm, is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells = run_algorithm(
                chars,
                pattern,
                options_steering,
                session_id,
                ask_user_confirmation=False,
                context_label=f"{pattern_id} [{idx}/{total_patterns}]",
            )
        except Exception as e:
            print()
            print(f"REQUEST FAILED  |  pattern_id={pattern_id}  |  request={idx}/{total_patterns}")
            print("ERROR: Could not generate a valid instance with the specified criteria..")
            print("Message:", str(e))
            print(traceback.format_exc())
            print()

            instances_and_solutions.append(None)
            continue

        print(
            f"REQUEST SUCCEEDED  |  pattern_id={pattern_id}  |  request={idx}/{total_patterns}  |  "
            f"hints={number_non_empty_cells}  |  weight={weight}",
        )
        if counts_techniques:
            print(f"Techniques used: {counts_techniques}")
        print()

        instances_and_solutions.append(
            (solution, instance, is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells)
        )



    # Make sure we haven't made a coding error
    assert len(instances_and_solutions) == len(patterns)

    # 4 Write output to file
    output_file_name = f"output_patterns_list_{session_id}.xlsx"
    print()
    print("Write output to file:", output_file_name)
    try:
        write_list_of_patterns_to_excel_file(file_name_pattern, output_file_name, patterns, instances_and_solutions)
    except Exception as e:
        print("Something went wrong when trying to write output to file:", str(e))
        print(traceback.format_exc())

    # 5 No user logs generated

    # 6 No algorithm logs generated


def run_for_single_pattern(chars, pattern, options_steering, session_id, file_name_template):

    print("Pattern:")
    for row in pattern:
        print(["o" if e else " " for e in row])
    print()

    # 2 Algorithm

    solution, instance, logs_algorithm, is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells = run_algorithm(
        chars, pattern, options_steering, session_id, ask_user_confirmation=True
    )

    # 4 Write output to file
    file_name_output_base = f"output_patterns_single_{session_id}"
    output_file_name = file_name_output_base + ".xlsx"
    print()
    print("Write output to file:", output_file_name)
    try:
        write_to_excel_file(
            output_file_name, pattern, solution, instance,
            is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells,
        )
    except Exception as e:
        print("Something went wrong when trying to write output to file:", str(e))
        print(traceback.format_exc())

    # 5 Generate step-by-step logs for the final instance
    try:
        generate_logs(instance, output_file_name)
    except Exception as e:
        print("Something went wrong when trying to generate logs for solving the final instance using human techniques:", str(e))

    # 6 Generate algorithm logs
    # Note: The logs are only written to file if a template file is provided as an optional command line argument
    if file_name_template is not None:

        file_name_output_logs_generator = file_name_output_base + "_logs_algorithm" + ".xlsx"
        print(f"Write algorithm logs to file:", file_name_output_logs_generator)
        try:
            instance_details = (instance, is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells)
            write_logs_to_file(pattern, chars, instance_details, logs_algorithm, file_name_template, file_name_output_logs_generator)
        except Exception as e:
            print("Something went wrong when trying to generate algorithm logs:", str(e))
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
