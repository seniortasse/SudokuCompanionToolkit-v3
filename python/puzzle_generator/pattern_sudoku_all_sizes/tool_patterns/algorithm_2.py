

# The new version of the algorithm to find uniquely solvable instances with hints according to a pre-specified pattern,
#  which fills in values for the "black squares" one-by-one and from left-to-right, while using constraint propagation
#  to determine at each step whether the instance is still (likely, as not all constraints are implemented, but
#  empirically this is about ~90% accurate) solvable, and specifically keeping a collection of possible values for each
#  cell to be chosen from,
#  using a backtracking
#  mechanism to return to a previous state in case we find out later that the current state is actually not valid
#  (which can happen since not all constraints are taken into consideration, which would be a massive undertaking,
#  instead currently only the first-order constraints concerning dimensions are implemented, which can be improved by
#  implementing more constraints, which can be reused from the already implemented human techniques)

# Algorithm steps:
#  - Fill in new values one-by-one according to the black squares of the pattern
#  - Propagate constraints
#  - Determine whether the instance is still (likely) solvable - If so, continue, if not, use backtracking to return to
#    a previous state which was still (likely) solvable
#  - Until all values for the black squares of the pattern are filled in
#  - Repeat until an instance is found which is uniquely solvable

# Algorithm settings:
#  - Running for a specified duration (15 minutes) until an instance is found with a unique solution

# User input configuration:
#  - When a valid solution is found, the user is asked to accept/reject the solution (based on the reported required
#    implemented human techniques, and the corresponding weight)
#  - After the specified duration has been reached without finding an accepted valid solution, the user is asked to
#    continue or abort the search


from collections import Counter
import itertools
import random
import time

from generator.model import Instance, count_non_empty_cells
from generator.verification import solve, check_is_unique_solution, NoSolutionException, verify_base_constraints
from generator.algo_human import solve_using_human_techniques, TECHNIQUES, TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT
from generator.weights import determine_weight

from tool_patterns.algo_backtracking_propagation import EMPTY_CHAR, generate_grids_using_propagation
from tool_patterns.output import write_instances_to_file, write_debug_info_to_file


# TODO This is currently only checked after running an entire process, and should be checked inside the function to be
#  made more accurate
MAX_RUNTIME_MINUTES = 15


def run_algorithm(chars, pattern, options_steering, session_id, ask_user_confirmation=True, context_label=None):
    """
    Steps:
      1 - Generate an instance
      2 - Checks:
          - All characters present
          - At least one solution (done separately as the code for checking whether there is a unique solution requires
            that there is at least one solution)
          - A unique solution
          - Provided extra steering options
    """

    size = pattern.size

    time_start = time.time()

    debug_info = {}

    # Initialise the seed generator so that all random seeds will be the same, to reproduce a run exactly
    seed_seed_generator = random.randint(0, 100_000)
    # seed_seed_generator = 39115
    seed_generator = random.Random(seed_seed_generator)
    print("Using seed for seed generator:", seed_seed_generator)

    debug_info["seed"] = seed_seed_generator

    # Valid instances which do not satisfy the steering criteria will be added to this list, to be saved to file in case
    #  finally no instances satisfying those criteria is found and the user indicates to abort and save all instances to
    #  file
    all_instances = []

    # When generating an instance for a solution with a non-standard boxes layout, do not include the boxes-x
    #  techniques, as their implementation requires a standard boxes layout
    techniques_disabled = TECHNIQUES_REQUIRING_STANDARD_BOXES_LAYOUT if pattern.uses_custom_boxes_layout else []
    if techniques_disabled:
        print(f"Disabled techniques {techniques_disabled} as the solution uses a custom boxes layout")
    use_techniques = [technique for technique in TECHNIQUES if technique not in techniques_disabled]

    counter = 0
    while True:
        counter += 1

        time_now = time.time()
        current_runtime_seconds = time_now - time_start

        # Ask user whether to continue or abort after the specified maximum algorithm duration has expired
        if current_runtime_seconds >= MAX_RUNTIME_MINUTES * 60:

            if not ask_user_confirmation:
                # Raise an exception to be handled by the calling piece of code
                raise Exception(f"Could not generate a valid instance within {MAX_RUNTIME_MINUTES} minutes")

            print()
            while True:
                _input = input('\n'.join([
                    f"Could not find a solution within {MAX_RUNTIME_MINUTES} minutes:",
                    f" c - continue",
                    f" w - write all found instances to file and quit",
                    f" q - quit without writing instances to file",
                    "",
                ]))
                print("Input:", _input)
                if _input == "c":
                    print("Continuing..")
                    # Reset the time counter
                    time_start = time.time()
                    break
                elif _input == "w":
                    try:
                        output_file_name = f"all_instances_{session_id}.rtf"
                        write_instances_to_file(output_file_name, all_instances)
                    except Exception as e:
                        print("Something went wrong when trying to write all instances to file:", str(e))
                    quit_tool(debug_info, session_id)
                elif _input == "q":
                    print("Aborting..")
                    quit_tool(debug_info, session_id)
                else:
                    print("Input not recognised, please try again..")
                    continue

        

        elapsed_minutes = current_runtime_seconds / 60
        remaining_seconds = max(0.0, MAX_RUNTIME_MINUTES * 60 - current_runtime_seconds)
        remaining_minutes = remaining_seconds / 60
        progress_pct = min(100.0, (current_runtime_seconds / (MAX_RUNTIME_MINUTES * 60)) * 100.0)
        label_prefix = f"[{context_label}] " if context_label else ""

        print()
        print(
            f"{label_prefix}Try {counter} | "
            f"elapsed={elapsed_minutes:.2f} min | "
            f"remaining={remaining_minutes:.2f} min | "
            f"budget_used={progress_pct:.1f}%"
        )

        # For each try the random generator is re-initialised with a random seed; Used to be able to re-run for
        #  debugging purposes (random costs are used in the solver, as a speedup technique)
        seed = seed_generator.randint(0, 100_000)
        random.seed(seed)
        print("Using random seed:", seed)

        # To avoid getting stuck for a long time in an unpromising partial instance, we re-initialise the search after
        #  every found solution
        grid_generator = generate_grids_using_propagation(chars, pattern, show_logs=False)
        try:
            flat_grid, logs = next(grid_generator)
            assert isinstance(flat_grid, list)
            assert isinstance(logs, dict)
        except StopIteration:
            print("Could not find a solution, investigate..")
            continue

        # Post-process the grid: Remove propagated values on the white squares and unflatten
        _flat_grid = [flat_grid[idx] if pattern[idx // size][idx % size] else EMPTY_CHAR for idx in range(size ** 2)]
        grid = [_flat_grid[size * i:size * (i + 1)] for i in range(size)]

        print("Found instance:")
        for row in grid:
            print(row)

        print("Performing checks...")

        # Check 1: All characters present
        print("Check 1: All characters present")
        all_chars_present = len(set(chars).difference(set(itertools.chain(*grid)))) == 0
        if not all_chars_present:
            print("Not all characters present in instance, trying again..")
            continue
        else:
            print(Counter(e for row in grid for e in row if e != EMPTY_CHAR))

        # Note: Can only create an instance when all characters are present, otherwise an exception is thrown
        # It seems that this is the only place where we create an Instance, so we can define some custom logic here
        instance = Instance(grid, preprocessed_dims=pattern.dims)

        # Check 2: At least one solution
        print("Check 2: At least one solution")
        try:
            solution = solve(instance)
        except NoSolutionException:
            print("No solution for instance, trying again..")
            continue
        else:
            pass

        # Check 3: Unique solution
        print("Check 3: A unique solution")
        # TODO Ideally, we should only sole the instance once to determine whether there is a unique solution (combine
        #  with whether there is at least once solution)
        has_unique_solution = check_is_unique_solution(instance, solution)
        if not has_unique_solution:
            print("No unique solution for instance, trying again..")
            continue
        else:
            print(solution)

        # Ask user for input to accept/reject the instance based on the techniques used and corresponding weight
        solved_instance_using_human_techniques, (counts_techniques, _) = solve_using_human_techniques(
            instance, use_techniques=use_techniques
        )
        is_solvable_using_human_techniques = solved_instance_using_human_techniques == solution
        print()
        print("Instance solvable using human techniques:", is_solvable_using_human_techniques)
        weight = None
        if is_solvable_using_human_techniques:
            print("Techniques used:", counts_techniques)
            weight = determine_weight(counts_techniques)
            print("Weight:", weight)
        print()

        # Here we already want to store the instance to report later in case no instance could be found satisfying
        #  the steering criteria
        all_instances.append((instance, counts_techniques, weight))

        # Check 4: Instance satisfies the provided steering criteria

        # Note: When checking steering options, the instance has to be solvable with human techniques, otherwise the
        #  criteria are meaningless
        if len(options_steering) > 0 and (not is_solvable_using_human_techniques):
            print("Instance not solvable so cannot validate the steering criteria, try again..")
            continue

        target_weight_range = options_steering.get("-w")
        if target_weight_range is not None:
            min_weight, max_weight = target_weight_range
            if not min_weight <= weight <= max_weight:
                print("Try again as the instance does not satisfy the steering option for weight range:", target_weight_range)
                continue

        techniques_used = set(counts_techniques.keys())

        target_required_techniques = options_steering.get("-t")
        if target_required_techniques is not None:
            techniques_unused = set(target_required_techniques).difference(techniques_used)
            all_required_techniques_used = len(techniques_unused) == 0
            if not all_required_techniques_used:
                print("Try again as the instance does not satisfy the steering option for techniques required:", techniques_unused)
                continue

        # Exclude techniques if indicated by the inputs
        techniques_excluded = options_steering.get("-e")
        if techniques_excluded is not None:
            excluded_techniques_used = set(techniques_used).intersection(techniques_excluded)
            no_excluded_techniques_used = len(excluded_techniques_used) == 0
            if not no_excluded_techniques_used:
                print("Try again as the instance does not satisfy the steering option for excluded techniques:", excluded_techniques_used)
                continue

        target_number_techniques_range = options_steering.get("-n")
        if target_number_techniques_range is not None:
            num_techniques_used = len(techniques_used)
            min_number_techniques, max_number_techniques = target_number_techniques_range
            if not min_number_techniques <= num_techniques_used <= max_number_techniques:
                print("Try again as the instance does not satisfy the steering option for number different techniques used:", target_number_techniques_range)
                continue

        print("Instance satisfies all requirements!")

        if not ask_user_confirmation:
            # Accept the generated solution/instance immediately
            break

        is_accepted = False
        while True:
            _input = input(f"Accept (a) or reject (r) this solution, or quit (q)? ")
            print("Input:", _input)
            if _input == "a":
                print("Accepted..")
                is_accepted = True
                break
            elif _input == "r":
                print("Rejected, trying again..")
                break
            elif _input == "q":
                print("Aborting..")
                quit_tool(debug_info, session_id)
            else:
                print("Input not recognised, please try again..")
                continue
        if is_accepted:

            # Also write debug info to file in case a solution was found (but only if user confirmation is enabled)
            write_debug_info_to_file(debug_info, session_id)

            break

    # TODO Final verifications using tested functions on the proposed instance

    # TODO This can be removed, for now included for compatibility with algorithm 1
    # subwords_placements = []
    # subwords_not_included = []

    # Determine whether the accepted instance is solvable using the implemented human techniques
    solved_instance_using_human_techniques, (counts_techniques, _) = solve_using_human_techniques(
        instance, use_techniques=use_techniques
    )
    is_solvable_using_human_techniques = solved_instance_using_human_techniques == solution
    print()
    print("Instance solvable using human techniques:", is_solvable_using_human_techniques)
    if is_solvable_using_human_techniques:
        print("Techniques used:", counts_techniques)

    # Extra information about instance
    weight = determine_weight(counts_techniques)
    number_non_empty_cells = count_non_empty_cells(instance)

    return solution, instance, logs, is_solvable_using_human_techniques, counts_techniques, weight, number_non_empty_cells


def quit_tool(debug_info, session_id):
    """
    Wrapper for quitting the tool, by first performing some steps, such as writing debug info to file
    """

    # First write the debug info to file
    write_debug_info_to_file(debug_info, session_id)

    quit()
