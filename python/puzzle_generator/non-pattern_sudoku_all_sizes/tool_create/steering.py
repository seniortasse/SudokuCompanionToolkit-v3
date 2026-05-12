
import random
import time

from generator.generator import generate_instance
from generator.model import count_non_empty_cells
from generator.algo_human import TECHNIQUES, TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT
from generator.weights import WEIGHTS, determine_weight

from tool_create.default_steering_parameters import DEFAULT_STEERING_PARAMETERS


RATINGS = [1, 2, 3, 4]


# Functionality:
#  - Overwriting target ranges if specified in the steering options
#  - Storing all otherwise valid instances that do not satisfy the steering options, optionally to be written to file
#    later
#  - Ask for user input when after a certain number of tries or minutes no instance is found that satisfies the steering
#    options


def generate_instance_with_rating(solution, rating, options_generator, options_steering, all_instances, max_number_tries=None, max_runtime_minutes=None, show_logs=False):
    assert rating in RATINGS, "Passed rating not valid"
    assert max_number_tries is None or max_number_tries > 0, "Max number of tries should be positive"
    assert max_runtime_minutes is None or max_runtime_minutes > 0, "Max runtime should be positive"

    size = solution.size
    assert size in DEFAULT_STEERING_PARAMETERS, f"Steering parameters for grid size {size} not defined"
    params_for_size = DEFAULT_STEERING_PARAMETERS[size]
    assert rating in params_for_size, f"Steering parameters for rating {rating} not defined"
    params_for_rating = params_for_size[rating]

    target_range_non_empty_cells = params_for_rating["non-empty-cells"]
    target_range_weight = params_for_rating["weight"]
    allowed_techniques = params_for_rating["techniques-allowed"]

    # When generating an instance for a solution with a non-standard boxes layout, do not include the boxes-x
    #  techniques, as their implementation requires a standard boxes layout
    techniques_disabled = TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT if solution.uses_custom_boxes_layout else []
    if techniques_disabled:
        print(f"Disabled techniques {techniques_disabled} as the solution uses a custom boxes layout")
    allowed_techniques = [technique for technique in allowed_techniques if technique not in techniques_disabled]

    # Extra check: For all allowed techniques, a weight is defined (as this might introduce errors after implementing
    #  new techniques without updating the weights)
    #  -> This is now done already for all techniques in the file containing the weights
    undefined_weights = set(allowed_techniques).difference(WEIGHTS.keys())
    assert not undefined_weights, f"No weights defined for techniques: {undefined_weights}"

    # Override weight range if specified in the input arguments
    target_range_weight = options_steering.get("-w", target_range_weight)

    # Define required techniques if specified in the input arguments
    target_required_techniques = options_steering.get("-t", [])

    # Define range of number of different techniques used if specified in the input arguments
    target_range_number_different_techniques = options_steering.get("-n")

    # Exclude techniques if indicated by the inputs -> filter out instances containing this technique, to keep the
    #  results with tool_logs consistent without having to manually exclude these techniques
    techniques_excluded = options_steering.get("-e", [])
    # if techniques_excluded is not None:
    #     allowed_techniques = [name for name in allowed_techniques if name not in techniques_excluded]

    # Initialise the stop condition counters
    number_tries = 0
    time_start = time.time()

    is_valid_instance = False
    while not is_valid_instance:
        number_tries += 1

        target_non_empty_cells = random.randint(*target_range_non_empty_cells)
        min_non_empty_cells = target_range_non_empty_cells[1]

        print("Searching with allowed techniques:", allowed_techniques)
        print(f"Aiming for: {target_non_empty_cells} non-empty cells")
        print()

        instance, (counts_techniques, logs) = generate_instance(
            solution, target_non_empty_cells=target_non_empty_cells, allowed_techniques=allowed_techniques,
            only_accept_human_solvable_instances=True, options=options_generator,
            show_logs=show_logs,
        )
        weight = determine_weight(counts_techniques)
        techniques_used = list(counts_techniques.keys())

        # Returned and logged to file in case a valid instance could not be found for the steering criteria
        all_instances.append((instance, counts_techniques, weight))

        # Criteria: Number of hints (this is the main steering criteria, given to the generator)
        number_non_empty_cells = count_non_empty_cells(instance)
        is_valid_number_non_empty_cells = number_non_empty_cells <= min_non_empty_cells

        # Criteria: Weight (this is a required steering criteria, but only checked after generating)
        min_weight, max_weight = target_range_weight
        is_valid_weight = min_weight <= weight <= max_weight

        # Optional criteria: Techniques required
        required_techniques_not_used = sorted(
            set(target_required_techniques).difference(techniques_used),  key=lambda item: TECHNIQUES.index(item)
        )
        is_valid_techniques_required = len(required_techniques_not_used) == 0

        # Optional criteria: Techniques excluded
        excluded_techniques_used = sorted(
            set(techniques_used).intersection(techniques_excluded), key=lambda item: TECHNIQUES.index(item)
        )
        is_valid_techniques_excluded = len(excluded_techniques_used) == 0

        # Optional criteria: Number of different techniques used
        num_different_techniques = len(counts_techniques.keys())
        if target_range_number_different_techniques is not None:
            min_num_different_techniques, max_num_different_techniques = target_range_number_different_techniques
            is_valid_number_different_techniques_used = \
                min_num_different_techniques <= num_different_techniques <= max_num_different_techniques
        else:
            is_valid_number_different_techniques_used = True

        print()
        if not is_valid_number_non_empty_cells:
            print(f"Try again as the number of non-empty cells is too large: {number_non_empty_cells} vs {target_non_empty_cells}")
        elif not is_valid_weight:
            print(f"Try again as the weight is not within the target range: {weight} vs {target_range_weight}")
        elif not is_valid_techniques_required:
            print(f"Try again as not all required techniques were used: {required_techniques_not_used}")
        elif not is_valid_techniques_excluded:
            print(f"Try again as some of the excluded techniques were used: {excluded_techniques_used}")
        elif not is_valid_number_different_techniques_used:
            print(f"Try again as the number of different techniques is not within the target range: {num_different_techniques} vs {target_range_number_different_techniques}")
        else:
            is_valid_instance = True

        # Stop condition 1: Max number tries
        if max_number_tries is not None and number_tries == max_number_tries:
            break

        # Stop condition 2: Max runtime minutes
        time_now = time.time()
        current_runtime_seconds = time_now - time_start
        if current_runtime_seconds >= max_runtime_minutes * 60:
            break

    print()
    print(f"Search criteria for rating {rating}:")
    print(" - Target number of hints:", target_range_non_empty_cells)
    print(" - Target weight:", target_range_weight)
    print(" - Allowed techniques:", allowed_techniques)
    if target_required_techniques:
        print(" - Required techniques:", target_required_techniques)
    if techniques_excluded:
        print(" - Excluded techniques:", techniques_excluded)
    if target_range_number_different_techniques is not None:
        print(" - Target number of different techniques:", target_range_number_different_techniques)

    if not is_valid_instance:
        raise Exception(f"Could not find a valid instance within {max_number_tries} tries and {max_runtime_minutes} minutes")

    return instance, counts_techniques, logs, weight, number_tries
