
from generator.algo_human import TECHNIQUES

from tool_create.steering import RATINGS
from tool_create.input import read_solution_from_excel_file, read_list_of_solutions_from_excel_file


class SyntaxInvalidException(Exception):
    pass


class InputsInvalidException(Exception):
    pass


VALID_RUN_MODES = ["--single", "--list"]

VALID_OPTIONS = ["-n", "-w", "-t", "-e", "-s", "-f"]

OPTIONS_S = ["center", "horizontal", "vertical", "diagonal-1", "diagonal-2"]


def process_inputs(args):
    """
    Validate input arguments (syntax & values) and read data from file.

    Raises a
      - SyntaxInvalidException: if the syntax of the input arguments are not correct
      - InputsInvalidException: if the data specified by the input arguments is not valid

    The following input arguments from the command line are expected:
      - arg1: file name of the Excel file specifying the solution to generate an instance for
      - arg2: the difficulty rating
      - args...: optional arguments specifying the steering options to be used in the algorithm
    """

    # Check number of arguments (note: the "help" command is filtered out already)
    try:
        assert len(args) > 0, "No inputs were provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    try:
        assert len(args) >= 3, "Incorrect number of arguments provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    options_steering = {}
    options_generator = {}

    file_name_template = None

    try:

        # Argument 1: Run mode
        run_mode = args[0]
        assert run_mode in VALID_RUN_MODES, f"Run mode not correctly specified, should be in {VALID_RUN_MODES}"

        # Argument 2: File name of solution
        file_name_input = args[1]
        assert file_name_input.endswith(".xlsx"), "Input file should be a .xlsx file"

        # Argument 3: Rating
        rating = int(args[2])
        assert rating in RATINGS, f"Rating should be in {RATINGS}"

        # Extra arguments: Options
        idx_args = 3
        while len(args) > idx_args:
            option = args[idx_args]
            assert option in VALID_OPTIONS, f"Option {option} not recognised"
            assert option not in options_steering, f"Option {option} specified multiple times"
            try:
                value = args[idx_args + 1]
            except IndexError:
                raise InputsInvalidException(f"No value provided for option {option}")
            if option in ["-n", "-w"]:
                assert value.count("-") == 1, f"Range not correctly specified for option {option}: {value}"
                rng = value.split("-")
                assert all(val.isdigit() for val in rng), f"Range not correctly specified for option {option}: {value}"
                rng_min, rng_max = map(int, rng)
                assert rng_min <= rng_max, f"Range max cannot be smaller than range min for option {option}"
                assert rng_max > 0, f"Range max should be positive for option {option}"
                options_steering[option] = (rng_min, rng_max)
            elif option in ["-t", "-e"]:
                techniques = value.replace(" ", "").split(",")
                for technique in techniques:
                    assert technique in TECHNIQUES, f"Technique not recognised: {technique}"
                assert len(techniques) == len(set(techniques)), "Some techniques were specified multiple times"
                options_steering[option] = techniques
            elif option in ["-f"]:
                assert value.endswith(".xlsx"), "Template file for removal logs should be a .xlsx file"
                allowed_run_modes = ["--single"]
                assert run_mode in allowed_run_modes, f"Can only create removal logs for run modes {allowed_run_modes}"
                file_name_template = value
            else:  # -s
                valid_values = OPTIONS_S
                assert value in valid_values
                options_generator[option] = value
            idx_args += 2

    except AssertionError as e:
        raise InputsInvalidException(str(e))

    solutions = []
    try:
        if run_mode == "--single":

            solution, subwords_placements, is_fixed_diagonal = read_solution_from_excel_file(file_name_input)
            solutions.append(("solution", solution, subwords_placements, is_fixed_diagonal))

            # As for now, cannot use the symmetry option when the diagonal or subwords should be removed
            # Note: This is only relevant for single solution input
            if options_generator.get("-s") is not None:
                assert not is_fixed_diagonal, "Cannot use symmetry when the diagonal is removed"
                assert len(subwords_placements) == 0, "Cannot use symmetry when there are subwords present"

        elif run_mode == "--list":
            solutions_from_file = read_list_of_solutions_from_excel_file(file_name_input)
            # Processing to use the same values as for a single solution input
            # Note: For now this is not used in the main code, but this can be in case the input format is updated (in
            #  this case the check that has been moved to single solution input only has to be enabled here as well)
            for solution_id, solution in solutions_from_file:
                subwords_placements, is_fixed_diagonal = [], False
                solutions.append((solution_id, solution, subwords_placements, is_fixed_diagonal))
        else:
            raise NotImplementedError(f"Reading solution(s) for run mode '{run_mode}' not implemented")
    except Exception as e:
        raise InputsInvalidException(str(e))

    return file_name_input, run_mode, solutions, rating, options_steering, options_generator, file_name_template
