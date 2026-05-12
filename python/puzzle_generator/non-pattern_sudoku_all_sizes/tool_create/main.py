
import os
import random
import sys
import time
import traceback

sys.path.append(os.getcwd())

from generator.model import count_non_empty_cells
from generator.algo_human import TECHNIQUES

from tool_create.parser import process_inputs, SyntaxInvalidException, InputsInvalidException, OPTIONS_S, VALID_RUN_MODES
from tool_create.positioner import get_idxs_for_subword_placement
from tool_create.steering import generate_instance_with_rating, RATINGS
from tool_create.output import write_to_excel_file, pretty_print, write_instances_to_file, write_list_of_solutions_to_excel_file
from tool_create.logs import generate_logs
from tool_create.logs_generator import write_logs_to_file


# Aggregation of functionality implemented for different grid sizes:
#  - Optionally fix main diagonal
#  - Optionally provide steering criteria
#  - Dynamic handling of grid layout
#  - Single steering logic for all grid sizes
#  - Same weights for all grid sizes
#  - All default steering parameters defined in file default_steering_parameters.py


# PARAMETERS

# The maximum number of tries the algorithm will perform to generate a valid instance
#  - This is to avoid any computing issues when the criteria are too difficult or impossible to achieve; Especially in
#    case of the smaller grids generating instances is very fast, so if the number of tries is not limited the tool
#    might affect the system
MAX_NUMBER_TRIES = 50

# The maximum number of minutes the algorithm will try to generate a valid instance
#  - Note: This is added as a second stop condition as for different grid sizes the time per try is very different
# TODO This is currently only checked after running an entire process, and should be checked inside the function to be
#  made more accurate
MAX_RUNTIME_MINUTES = 5


def main():
    """
    Steps:
      0 Process command
      1 Read & validate data from file
      2 Generate an instance for the provided solution and target rating, repeat until one is accepted by the user
      3 Write instance to file, including some extra information
      4 Generate logs with step-by-step overview of techniques used to solve the final instance
      5 Generate logs with step-by-step overview of the value-removal process in the instance generator
    """

    # Constants
    # Removed as the tool can now be used for all grid sizes

    # Syntax
    tool_name = f"tool_create"
    syntax = "\n ".join([
        "  python {}/main.py".format(tool_name) + " {run_mode} {file_name} {difficulty} {options}",
        "",
        "  run_mode:     {}".format(VALID_RUN_MODES),
        "  file_name:    {}".format(".xlsx"),
        "  difficulty:   {}".format(RATINGS),
        "",
        "  Options:",
        "   -n value1-value2 (range of number of different techniques used)",
        "   -w value1-value2 (range of weight)",
        "   -t \"technique1, technique2, ...\" (list of techniques required - spaces not necessary, quotes necessary)",
        "      available techniques: " + ", ".join(TECHNIQUES),
        "   -e \"technique1, technique2, ...\" (list of techniques excluded - spaces not necessary, quotes necessary)",
        "      available techniques: " + ", ".join(TECHNIQUES),
        "   -s {} (whether to create symmetry in the indicated direction)".format(OPTIONS_S),
        "   -f File name of template used to generate logs for value-removal process (.xlsx)",
    ])

    print()

    # 0 Process command line arguments
    args = sys.argv[1:]

    # Check for help command
    if len(args) == 1 and args[0] in ["-h", "--help"]:
        print("The following syntax is used:")
        print(syntax)
        quit()

    # 1 Read & validate input data
    try:
        # Note: Layout is read from file and this information is included in the solution object
        file_name_input, run_mode, solutions, rating, options_steering, options_generator, file_name_template = \
            process_inputs(args)
    except (SyntaxInvalidException, InputsInvalidException) as e:
        print()
        print("Syntax/Inputs invalid!")
        print(str(e))
        if isinstance(e, SyntaxInvalidException):
            print()
            message = '\n\n'.join(["Please provide input using the following syntax:", syntax])
            print(message)
        else:
            pass
        quit()

    print()
    print(f"Generating an instance for {len(solutions)} solution(s)")
    print(f"With rating: {rating}")
    print("And optional steering criteria:")
    option_names_steering = {
        "-n": "Number of different techniques used",
        "-w": "Weight",
        "-t": "Techniques required",
        "-e": "Techniques excluded",
    }
    for option_steering in options_steering.items():
        print(" %s: %s" % (option_names_steering[option_steering[0]], option_steering[1]))
    print("And generator options:")
    option_names_generator = {
        "-s": "Symmetry",
    }
    for option_generator in options_generator.items():
        print(" %s: %s" % (option_names_generator[option_generator[0]], option_generator[1]))
    print()

    # To make it easier to reproduce runs, always initialise the seed, which can furthermore by fixed when testing
    seed = random.randint(1, 100_000)
    # seed = 1
    random.seed(seed)
    print()
    print("Initialising tool with random seed:", seed)
    print()

    # The flow is significantly different for each run mode, so this is split in separate functions
    if run_mode == "--single":
        _, solution, subwords_placements, is_fixed_diagonal = solutions[0]
        run_for_single_solution(solution, subwords_placements, is_fixed_diagonal, rating, options_generator, options_steering, file_name_input, file_name_template)
    elif run_mode == "--list":
        run_for_list_of_solutions(solutions, rating, options_generator, options_steering, file_name_input)
    else:
        raise NotImplementedError(f"Generating instances with run mode '{run_mode}' not implemented")

    print("Finished")


def run_for_list_of_solutions(solutions, rating, options_generator, options_steering, file_name_input):

    # Note: The options_generator is not updated here, as there are no subwords or fixed digaonal

    # Do ask for user confirmation once, to review and confirm the number of instances and steering criteria
    while True:
        _input = input("Continue? [y/n] ")
        if _input == "y":
            break
        elif _input == "n":
            print("Aborting..")
            quit()
        else:
            print("Input not recognised, try again..")

    # Store the instance found for each solution, to be written to file at the end
    instances = []

    for solution_id, solution, subwords_placements, is_fixed_diagonal in solutions:

        print(f"Generating an instance for solution: {solution_id}")
        pretty_print(solution, subwords_placements, is_fixed_diagonal)

        try:
            time_start = time.time()
            all_instances = []  # This is not used here
            instance, counts_techniques, logs_generator, weight, number_tries = generate_instance_with_rating(
                solution, rating, options_generator, options_steering, all_instances,
                max_number_tries=MAX_NUMBER_TRIES, max_runtime_minutes=MAX_RUNTIME_MINUTES, show_logs=False
            )
            time_end = time.time()
        except Exception as e:
            print()
            print("ERROR: Could not generate a valid instance with the specified criteria..")
            print("Message:", str(e))
            print(traceback.format_exc())
            print()

            # No user input asked for continuing/aborting

            # Add an empty result and continue to the next solution
            instances.append(None)
            continue

        print()
        number_non_empty_cells = count_non_empty_cells(instance)
        print(f"Found instance for rating {rating} with:")
        print(" - Number of hints:", number_non_empty_cells)
        print(" - Weight:", weight)
        print(" - Techniques used:", counts_techniques)
        print("Search took %s seconds and %s tries" % (time_end - time_start, number_tries))

        print()
        print(instance)

        # No user input asked for accepting/rejecting the instance

        # Add the instance including some . to the list and continue to the next solution
        instances.append((instance, counts_techniques, weight, number_non_empty_cells))

    # Make sure we haven't made a coding error
    assert len(solutions) == len(instances)

    # 3 Write output to file
    output_file_name = file_name_input.split('.')[0] + f"_{rating}.xlsx"
    print("Write output to file:", output_file_name)
    try:
        write_list_of_solutions_to_excel_file(solutions, instances, file_name_input, output_file_name)
    except Exception as e:
        print("Something went wrong when trying to write output to file:", str(e))

    # 4 No user logs generated

    # 5 No logs for the removal process generated


def run_for_single_solution(solution, subwords_placements, is_fixed_diagonal, rating, options_generator, options_steering, file_name_input, file_name_template):

    # Prepare options to be used in generator -> now received from inputs
    # options_generator = {}

    # Whether the diagonal is removed is either defined in the solution input file, or asked as user input otherwise
    options_generator["empty-diagonal"] = is_fixed_diagonal

    # Determine which idxs to remove for subword placements
    remove_idxs = []
    for subword, orientation, idx in subwords_placements:
        subword_idxs = get_idxs_for_subword_placement(subword, orientation, idx)
        remove_idxs.extend(subword_idxs)
    options_generator["remove-idxs"] = remove_idxs

    print(f"Generating an instance for solution:")
    pretty_print(solution, subwords_placements, is_fixed_diagonal)
    print("With subwords:")
    for item in subwords_placements:
        print("", item)

    # 2 Generate instance
    all_instances = []
    is_accepted = False
    while not is_accepted:

        # Each time a solution is rejected, we select a new random seed so that issues can be reproduced more easily
        seed = random.randint(1, 100_000)
        # seed = 12218
        random.seed(seed)
        print()
        print("Using random seed:", seed)
        print()

        try:
            time_start = time.time()
            instance, counts_techniques, logs_generator, weight, number_tries = generate_instance_with_rating(
                solution, rating, options_generator, options_steering, all_instances,
                max_number_tries=MAX_NUMBER_TRIES, max_runtime_minutes=MAX_RUNTIME_MINUTES, show_logs=False
            )
            time_end = time.time()
        except Exception as e:
            print()
            print("ERROR: Could not generate a valid instance with the specified criteria..")
            print("Message:", str(e))
            print(traceback.format_exc())
            print()
            # Ask user whether to try again or abort the tool
            is_abort = False
            while True:
                _input = input("Try again (r), write all found instances to file and quit (w) or quit (q)? ")
                print("Input:", _input)
                if _input == "r":
                    pass
                elif _input == "w":
                    is_abort = True
                    try:
                        write_instances_to_file(file_name_input, all_instances)
                    except Exception as e:
                        print("Something went wrong when trying to write all instances to file:", str(e))
                elif _input == "q":
                    is_abort = True
                else:
                    print("Input not recognised, please try again..")
                    continue
                break
            if is_abort:
                print("Aborting tool..")
                quit()
            else:
                print("Trying again..")
                continue

        print()
        number_non_empty_cells = count_non_empty_cells(instance)
        print(f"Found instance for rating {rating} with:")
        print(" - Number of hints:", number_non_empty_cells)
        print(" - Weight:", weight)
        print(" - Techniques used:", counts_techniques)
        print("Search took %s seconds and %s tries" % (time_end - time_start, number_tries))

        print()
        print(instance)

        # Ask user to accept/reject the result
        is_input_recognised = False
        while not is_input_recognised:
            text = input((
                "Accept?\n"
                " a - accept, write to file\n"
                " r - reject, search again\n"
                " q - quit tool\n"
            ))
            print("Input:", text)
            if text == "a":
                is_accepted = True
                is_input_recognised = True
            elif text == "r":
                is_accepted = False
                is_input_recognised = True
            elif text == "q":
                print("Aborting tool..")
                is_input_recognised = True
                quit()
            else:
                print("Input not recognised, please try again..")

    # 3 Write output to file
    file_name_output_base = file_name_input.split('.')[0] + f"_{rating}"
    output_file_name = file_name_output_base + ".xlsx"
    print("Write output to file:", output_file_name)
    try:
        write_to_excel_file(instance, rating, counts_techniques, weight, number_non_empty_cells, output_file_name)
    except Exception as e:
        print("Something went wrong when trying to write output to file:", str(e))

    # 4 Generate step-by-step logs for the final instance
    try:
        generate_logs(instance, output_file_name)
    except Exception as e:
        print("Something went wrong when trying to generate logs for solving the final instance using human techniques:", str(e))

    # Note: Currently these logs are only written to file if a template file is provided as an optional command line
    #  argument
    if file_name_template is not None:

        # 5 Generate removal logs
        file_name_output_logs_generator = file_name_output_base + "_logs_removal" + ".xlsx"
        print(f"Write logs for removal process to file:", file_name_output_logs_generator)
        try:
            write_logs_to_file(None, solution, instance, logs_generator, file_name_template, file_name_output_logs_generator)
        except Exception as e:
            print("Something went wrong when trying to generate logs for removal process:", str(e))


if __name__ == "__main__":
    main()
