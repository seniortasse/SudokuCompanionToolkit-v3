
from generator.algo_human import TECHNIQUES
from generator.model import Instance
from generator.boxes import create_default_layout_boxes

from tool_patterns.input import read_pattern_from_file, read_list_of_patterns_from_excel_file, read_layout_boxes


class SyntaxInvalidException(Exception):
    pass


class InputsInvalidException(Exception):
    pass


VALID_RUN_MODES = ["--single", "--list"]

VALID_SIZES = [4, 6, 8, 9, 16]

SIZES_REQUIRING_LAYOUT = [6, 8]

VALID_LAYOUTS = {
    6: ["3x2", "2x3"],
    8: ["4x2", "2x4"],
}

VALID_OPTIONS = ["-n", "-w", "-t", "-e", "-f", "-b"]


def process_inputs(args):
    """
    Validate input arguments (syntax & values) and read data from file if necessary.

    Raises a
      - SyntaxInvalidException: if the syntax of the input arguments are not correct
      - InputsInvalidException: if the data specified by the input arguments is not valid

    For this tool, only the single inputs from command line option is provided, without additional options:
      - arg1: characters + subwords
      - arg2: file name of the Excel file specifying the pattern
    """

    try:
        assert len(args) > 0, "No arguments were provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    try:
        assert len(args) >= 4, "Incorrect number of arguments provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    try:

        idx_args = 0

        # Argument 1: size & layout
        arg_size_and_layout = args[idx_args]
        size, layout = parse_size_and_layout(arg_size_and_layout)

        idx_args += 1

        # Argument 2: Characters + subwords
        chars_and_subwords = args[idx_args]
        assert chars_and_subwords.replace(' ', '').replace(',', '').isalnum(), "The first input argument should be a comma-separated list of alpha-numeric list of characters + subwords"

        words = chars_and_subwords.replace(' ', '').split(',')
        chars = words[0]
        subwords = words[1:]
        assert len(subwords) == 0, "This version of the tool is not able to handle subwords, please remove.."

        _validate_inputs(chars, subwords, size)

        # instance = (chars, subwords)

        idx_args += 1

        # Argument 3: Run mode
        run_mode = args[idx_args]
        assert run_mode in VALID_RUN_MODES, f"Run mode not correctly specified, should be in {VALID_RUN_MODES}"

        idx_args += 1

        # Argument 4: File name of pattern
        file_name_pattern = args[idx_args]
        assert file_name_pattern.endswith(".xlsx"), "Input file with the pattern(s) should be a .xlsx file"

        idx_args += 1

        # Optional arguments
        options_steering = {}
        file_name_template = None
        file_name_layout_boxes = None

        if len(args) > idx_args:
            options_steering, file_name_template, file_name_layout_boxes = parse_option_arguments(args, idx_args, run_mode)

        # After processing all arguments: Read from files

        patterns = []
        if run_mode == "--single":
            pattern = read_pattern_from_file(file_name_pattern, size)

            layout_boxes = None
            if file_name_layout_boxes is not None:
                layout_boxes = read_layout_boxes(file_name_layout_boxes, size)

            # Post-processing: Instantiate pattern to be able to reuse the preprocessed idxs throughout the algorithms
            # Note: This is done for an input list inside the read function
            pattern = Instance(pattern, is_chars=False, layout=layout, layout_boxes=layout_boxes)

            patterns.append(("pattern", pattern))
        elif run_mode == "--list":
            assert file_name_layout_boxes is None  # This is already checked when parsing arguments
            patterns = read_list_of_patterns_from_excel_file(file_name_pattern)
        else:
            raise NotImplementedError(f"Reading pattern(s) for run mode '{run_mode}' not implemented")

        # Check that the instantiation has been done properly
        for _, pattern in patterns:
            assert isinstance(pattern, Instance)

    except (AssertionError, Exception) as e:
        raise InputsInvalidException(str(e))

    return run_mode, file_name_pattern, chars, patterns, options_steering, file_name_template


def parse_size_and_layout(arg_size_and_layout):

    size_and_layout = arg_size_and_layout.split('-')
    assert len(size_and_layout) <= 2, "Size and layout not correctly specified"

    size = size_and_layout[0]
    assert size.isdigit(), "Size not correctly specified"
    size = int(size)
    assert size in VALID_SIZES, f"Size not valid, should be in {VALID_SIZES}"

    if len(size_and_layout) == 2:
        assert size in SIZES_REQUIRING_LAYOUT, f"Layout does not need to be specified for grid size {size}"
        layout = size_and_layout[1]
        valid_layouts = VALID_LAYOUTS[size]
        assert layout in valid_layouts, f"Layout not correctly specified, should be in {valid_layouts}"
    else:
        layout = None

    return size, layout


def parse_option_arguments(args, current_idx_args, run_mode):

    options_steering = {}

    file_name_template = None
    file_name_layout_boxes = None

    while len(args) > current_idx_args:

        # Parse name
        option = args[current_idx_args]
        assert option in VALID_OPTIONS, f"Option {option} not recognised"
        assert option not in options_steering, f"Option {option} specified multiple times"

        try:
            value = args[current_idx_args + 1]
        except IndexError:
            raise InputsInvalidException(f"No value provided for option {option}")

        # Parse value
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
            assert value.endswith(".xlsx"), "Template file for algorithm logs should be a .xlsx file"
            allowed_run_modes = ["--single"]
            assert run_mode in allowed_run_modes, f"Can only create algorithm logs for run modes {allowed_run_modes}"
            file_name_template = value
        elif option == "-b":
            # File name of file containing custom layout for boxes
            allowed_run_modes = ["--single"]
            assert run_mode in allowed_run_modes, f"Can only specify a custom layout for boxes for run modes {allowed_run_modes}"
            assert value.endswith(".xlsx"), "Input file specifying custom layout for boxes should be a .xlsx file"
            file_name_layout_boxes = value
        else:
            raise NotImplementedError(f"Input parsing for option {option} not defined")

        current_idx_args += 2

    return options_steering, file_name_template, file_name_layout_boxes


def _validate_inputs(chars, subwords, size):
    assert isinstance(chars, str)

    # Correct number of characters
    number_chars = len(chars)
    assert number_chars == size, f"Number of characters incorrect: {number_chars}"

    # Unique characters
    assert len(chars) == len(set(chars)), f"List of characters contains duplicates: \"{chars}\""

    # Subwords use valid characters
    MIN_LENGTH_SUBWORDS = 2
    for subword in subwords:
        assert len(set(subword).difference(chars)) == 0, f"Subword contains characters not present in the list of characters: \"{subword}\""
        assert len(subword) <= size, f"Subword too long: \"{subword}\""
        assert len(subword) >= MIN_LENGTH_SUBWORDS, f"Subword too short (min length: {MIN_LENGTH_SUBWORDS}): \"{subword}\""

    # No duplicated subwords
    assert len(subwords) == len(set(subwords)), f"List of subwords contains duplicates: {subwords}"
