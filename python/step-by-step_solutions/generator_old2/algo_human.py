
from collections import defaultdict
import colorama
from functools import partial
import random
import time

from generator.model import EMPTY_CHAR, count_empty_cells, get_box, copy_instance, Instance
from generator.techniques.singles import find_singles_1, find_singles_2, find_singles, find_naked_singles_2, find_naked_singles_3
from generator.techniques.multiples import find_doubles, find_triplets, find_quads
from generator.techniques.options import determine_options_per_cell, identify_new_values, copy_options
from generator.techniques.x_wings import find_x_wings
from generator.techniques.singles_pointing import find_pointing_singles
from generator.techniques.remote_pairs import find_remote_pairs
from generator.techniques.ab_chain import find_ab_chains
from generator.techniques.x_wings_multiple import find_x_wings_3, find_x_wings_4
from generator.techniques.singles_boxed import find_singles_boxed
from generator.techniques.y_wings import find_y_wings
from generator.techniques.boxed_wings import find_boxed_wings
from generator.techniques.boxed_multiples import find_boxed_doubles, find_boxed_triplets, find_boxed_quads
from generator.techniques.multiples_naked import find_naked_doubles, find_naked_triplets, find_naked_quads
from generator.techniques.ab_rings import find_ab_rings
from generator.techniques.boxed_rays import find_boxed_rays


colorama.init(autoreset=True)


debug_mode = True


# Used for profiling
times = defaultdict(float)
counts = defaultdict(int)


# Note: This also defines the order in which techniques are applied
TECHNIQUES = [
    # Singles-1: Some dimension has only 1 empty cell (can both be found by regular/naked singles technique)
    # Regular singles: A specific value can only be filled in some dimension by looking at the values in 1/2 other
    #  dimensions
    # Naked singles: For some cell, 2/3 dimensions exclude all values but one
    "singles-1",
    "singles-2", "singles-naked-2",
    "singles-3", "singles-naked-3",
    "doubles-naked",
    "triplets-naked",
    "quads-naked",
    "singles-pointing",
    "singles-boxed",
    "doubles",
    "triplets",
    "quads",
    "x-wings",
    "y-wings",
    "remote-pairs",
    "boxed-doubles",
    "boxed-triplets",
    "boxed-quads",
    "boxed-wings",
    "boxed-rays",
    "ab-rings",
    "ab-chains",
    "x-wings-3",
    "x-wings-4",
]


# Note: On purpose this is implemented as a set (which is an unordered collection), to make clear that the order is
#  defined in TECHNIQUES, and this is just a collection of values
BASE_TECHNIQUES = {
    "singles-1", "singles-2", "singles-3",
    "singles-naked-2", "singles-naked-3",
}

ADVANCED_TECHNIQUES = [name for name in TECHNIQUES if name not in BASE_TECHNIQUES]


fncs = {
    "singles-1": find_singles_1,
    "singles-2": find_singles_2,
    "singles-3": find_singles,
    "singles-naked-2": find_naked_singles_2,
    "singles-naked-3": find_naked_singles_3,
    "doubles-naked": find_naked_doubles,
    "triplets-naked": find_naked_triplets,
    "quads-naked": find_naked_quads,
    "singles-pointing": find_pointing_singles,
    "singles-boxed": find_singles_boxed,
    "doubles": find_doubles,
    "triplets": find_triplets,
    "quads": find_quads,
    "x-wings": find_x_wings,
    "y-wings": find_y_wings,
    "remote-pairs": find_remote_pairs,
    "boxed-doubles": find_boxed_doubles,
    "boxed-triplets": find_boxed_triplets,
    "boxed-quads": find_boxed_quads,
    "boxed-wings": find_boxed_wings,
    "boxed-rays": find_boxed_rays,
    "ab-rings": find_ab_rings,
    "ab-chains": find_ab_chains,
    "x-wings-3": find_x_wings_3,
    "x-wings-4": find_x_wings_4,
}

# TODO It would be better to define the list once
assert not set(TECHNIQUES).symmetric_difference(fncs.keys())


def solve_using_human_techniques(instance, use_techniques=TECHNIQUES,
                                 # Temporary indicator for which method to use: Regular or cleanup
                                 use_cleanup_method=True,
                                 include_magic_technique=False, magic_solution=None,
                                 max_number_iterations=None, show_logs=False):
    """
    Algorithm steps:
      While the instance is not entirely filled in and the max number of iterations not exceeded:
       - Apply the base techniques (singles) from easy to difficult in the order defined in BASE_TECHNIQUES, which looks
         directly at the instance and does not use options logic
       - If none of the base techniques could fill in a new value, apply the advanced techniques (all techniques defined
         in TECHNIQUES and not in BASE_TECHNIQUES) from easy to difficult; Each advanced technique might apply a
         combination of removing options and finding values directly (ALTHOUGH THIS SHOULD PROBABLY BE DONE AFTERWARDS),
         the reduced options are used for the next more difficult technique
       - TODO New option removal process
       - If none of the previous steps leads to finding a new value, the instance is considered unsolvable with the
         implemented human techniques; If the arguments indicate to include the magic technique, an empty cell is
         randomly picked and filled with the oracle value from the final solution, which is known and also given as an
         argument

    This function might return a partially filled solution, in case:
      - None of the implemented techniques could find a new value, and we are not using the magic technique
      - The max number of iterations is smaller than the initial number of empty cells
    This is perfectly acceptable, and even used in some functions using this function to check whether the instance is
    humanly solvable.
    """

    assert isinstance(instance, Instance)

    size = instance.size
    chars = instance.chars

    # Important, or some of the techniques will fail, as they assume all chars are contained in this set
    #  -> Possibly need to filter out such cases before calling this function
    assert len(chars) == size, f"{chars}"

    from collections import defaultdict
    counts_techniques = defaultdict(int)

    if show_logs:
        print("Instance:")
        print(instance)
        print()

    solved_instance = copy_instance(instance)

    # Currently added to logs:
    #  - steps: a list of instances with newly filled values and technique used
    logs = {}
    logs["steps"] = []

    step = 0
    is_filled = count_empty_cells(instance) == 0
    while not is_filled:
        step += 1

        # Second stop-condition of the while-loop
        if max_number_iterations is not None and step > max_number_iterations:
            break

        if show_logs:
            print(f"Step {step}")

        new_idxs = []

        # Used for logging
        log_new_values_found = None
        log_technique_used = None
        # Temporary while the cleanup process is not yet implemented
        log_is_cleanup_issue = False
        # TODO Now details are only added when a new value is found, we should also add details when options are removed
        log_cleanup_steps = []
        log_solved_instance_before = copy_instance(solved_instance)

        # Step 1: Basic techniques
        if show_logs:
            print("Apply the base (singles) techniques")
        could_use_technique = False
        for name in TECHNIQUES:
            if name in BASE_TECHNIQUES and name in use_techniques:

                if show_logs: print(format_bold(f"Identify {name}"))
                time_start = time.time()
                fnc = fncs[name]
                hits = fnc(solved_instance, show_logs=show_logs)
                time_end = time.time()
                times[name] += time_end - time_start
                counts[name] += 1
                if show_logs: print(f" Found {len(hits)} {name}: %s" % [tuple(e + 1 for e in t[0]) + t[1:] for t in hits])

                if len(hits) > 0:
                    counts_techniques[name] += len(hits)
                    new_idxs.extend([t[0] for t in hits])
                    for (i1, i2), char, dimension in hits:
                        solved_instance[i1][i2] = char
                    could_use_technique = True
                    # TODO Refactor code
                    log_new_values_found = hits
                    log_technique_used = name
                    break

        # Step 2: Advanced techniques
        # Current logic:
        #  - Determine initial options per cell by looking at occurrences of other values in row/col/box; This should
        #    represent the combination of all base techniques; Note that this is the case looking at how options are
        #    used:
        #    - In determine_options() possible values for each cell are determined by looking at other values in all
        #      dimensions (this is the naked-singles step - if this leads to a cell with only 1 option this is a
        #      naked-singles value)
        #    - In identify_new_values() (which is called after removing options with the advanced technique), we
        #      identify new values by looking at the cell (which is a naked-singled as described above), and row/col/box
        #      dimensions (which are singles-3, which generalise singles-1 and singles-2)
        #    Note also that by default we check that this does not yield any new values before applying any advanced
        #    techniques, as we have just applied the base techniques
        #  - Each advanced technique applied a combination of removing options and identifying new values based on all
        #    singles (base) techniques; Eg,
        #    - doubles/triplets/quads removes all options first and applied identify_new_values once at the end
        #    - the next techniques combine removing options and identifying new values within the algorithm
        #    - ab-chains does not remove any options but identifies a new value directly
        #    TODO This can be rewritten by only removing options in the functions, and applying identify_new_values()
        #     once after the function is called - but for now this would take too much time rewriting and testing
        if not could_use_technique and not use_cleanup_method:

            # raise Exception("Should use cleanup method")

            if show_logs:
                print("Apply advanced techniques iteratively as no new value could be found with the base techniques")

            # Options is determined once based on all singles techniques, and given as argument to advanced techniques,
            #  as it should be iteratively applied
            options = determine_options_per_cell(solved_instance)

            options_before = options

            if debug_mode:
                # Temporary verification step: After applying the basic techniques, it should not be possible to find
                #  any new values from the remaining options
                if not BASE_TECHNIQUES.difference(use_techniques):
                    _values = identify_new_values(options, chars, show_logs=show_logs)
                    assert len(_values) == 0

            for name in TECHNIQUES:
                if name not in BASE_TECHNIQUES and name in use_techniques:

                    if show_logs: print(f"Identify {name}")
                    time_start = time.time()
                    number_removed_values, hits, options, details = apply_technique(name, options, chars, solved_instance, show_logs=show_logs)
                    log_cleanup_steps.append((name, name, 0, options_before, details))
                    time_end = time.time()
                    times[name] += time_end - time_start
                    counts[name] += 1
                    # identify_new_values() might be called multiple times, we have to make sure only unique hits are
                    #  processed
                    # TODO Cannot identify_new_values() be called after removing options, and isn't this much cleaner?
                    #  -> Currently, it is only called one time after doubles/triplets/quads techniques, for other
                    #     techniques it is called multiple times for logging purposes (to log where the value was
                    #     found), whereas for ab-chains it is not called at all as the identifying logic is baked into
                    #     the search algorithm itself); We could rewrite the code for each of the advanced techniques
                    #     to only remove options, and call identify_new_values() once after the function, but for now
                    #     this would require too much rewriting and testing time

                    if len(hits) > 0:
                        assert len(hits) == 1

                        counts_techniques[name] += len(hits)
                        new_idxs.extend([t[0] for t in hits])

                        # for (i1, i2), char in hits:
                        for hit in hits:
                            # TODO Later assert that length is 3 when all techniques have details
                            assert 2 <= len(hit) <= 4
                            (i1, i2) = hit[0]
                            char = hit[1]
                            solved_instance[i1][i2] = char
                            options[i1][i2] = set()
                        could_use_technique = True
                        # TODO Refactor code
                        log_new_values_found = hits
                        log_technique_used = name
                        break

        # Step 3: Extra iterative option removal phase ("cleanup")
        #  - Instead of applying the advanced techniques once from easy to difficult, this is a more elaborate/complete
        #    process getting the full potential out of the implemented techniques by combining the option removal and
        #    value identification of each technique;
        #    In this process easier techniques are revisited after applying a more difficult technique;
        #    The process is as follows:
        #     - Start with applying the first advanced technique
        #       (note: we can skip the first one as we know from step 2 that this does not lead to any new value; It
        #       might remove options, but this will be found in the next step -> X this is invalid, as in case no value
        #       is found and no option removed we continue with the next technique without revisiting the first
        #       technique)
        #     - There are 3 possible outcomes:
        #       - A new value is found -> Abort and count this technique
        #       - No new value is found, but at least one option is removed -> Start the iterative subprocess
        #       - No new value is found and no option is removed -> Continue with the next advanced technique (we have
        #         already verified that no more options can be removed and no new value be found when iteratively
        #         applying the previous techniques - since this technique does not contribute anything, we do not need
        #         to revisit the earlier techniques)
        #    The iterative subprocess:
        #      - Apply all currently included techniques (in any order, but it makes the most sense to do this from easy
        #        to difficult); After applying one of these techniques, there are 3 possible outcomes:
        #        - A new value is found -> Abort and count the most difficult currently included technique (the one last
        #          added in the main process)
        #        - No new value is found (regardless of whether an option was removed) -> Continue applying all the
        #          currently included techniques
        #        - If at the end of the subprocess no new value was found, but at least one option was removed, apply
        #          all currently included techniques again (since the initial options are not the same, this time this
        #          might lead to finding a new value)
        #        - If at the end of the subprocess no new value was found and no options removed, we have exploited the
        #          full potential of all currently included techniques to no avail, and should add the next advanced
        #          technique to this collection, and start the process again

        # TODO It could even be helpful to apply a single technique iteratively - after the first round of
        #  removing options, the next iteration might be able to find a new value, as we have seen once when
        #  applying the doubles technique twice in a row! So don't continue here after the first iteration

        if not could_use_technique and use_cleanup_method:

            if show_logs:
                print("Starting the iterative \"cleanup\" process as the regular process could not find any values")

            # Similarly to step 2, we initialise the options, which implies applying all the base (singles) techniques
            options = determine_options_per_cell(solved_instance)

            # Since step 2 is done by default, we do not repeat the verification check that no new value can be found
            # based on the base techniques only -> now there is a switch for step 2/3
            if debug_mode:
                # Temporary verification step: After applying the basic techniques, it should not be possible to find
                #  any new values from the remaining options
                if not BASE_TECHNIQUES.difference(use_techniques):
                    _values = identify_new_values(options, chars, show_logs=show_logs)
                    assert len(_values) == 0

            try_techniques = [technique for technique in ADVANCED_TECHNIQUES if technique in use_techniques]

            included_techniques = []
            for technique_latest in try_techniques:
                if show_logs:
                    print(format_bold(f"Add next advanced technique ({technique_latest}) to the process"))
                included_techniques.append(technique_latest)

                options_before = options  # Note: This gets copied in the apply_technique function

                # First, apply the latest technique
                if show_logs:
                    print(format_bold(f"First try to apply the latest technique: {technique_latest}"))
                time_start = time.time()
                number_removed_values, hits, options, details = apply_technique(technique_latest, options, chars, solved_instance, show_logs=show_logs)
                log_cleanup_steps.append((technique_latest, technique_latest, 0, options_before, details))
                time_end = time.time()
                times[technique_latest] += time_end - time_start
                counts[technique_latest] += 1

                # For the first technique, just remove the options, as we know there cannot be a hit -> only if step 2 is enabled!
                # if try_techniques.index(technique) == 0:
                #     assert len(hits) == 0
                #     continue

                if len(hits) > 0:
                    assert len(hits) == 1

                    counts_techniques[technique_latest] += len(hits)
                    new_idxs.extend([t[0] for t in hits])
                    for hit in hits:
                        assert 2 <= len(hit) <= 4
                        (i1, i2) = hit[0]
                        char = hit[1]
                        solved_instance[i1][i2] = char
                        options[i1][i2] = set()
                    could_use_technique = True
                    # TODO Refactor code
                    log_new_values_found = hits
                    log_technique_used = technique_latest
                    break

                elif options != options_before:
                    if show_logs:
                        print("Start the subprocess as no new value was found but at least one option removed")

                    found_hit = False

                    counter_subprocess = 0
                    while True:
                        counter_subprocess += 1
                        if show_logs:
                            print(f"Iteration {counter_subprocess} of the subprocess including techniques: {included_techniques}")

                        # Update options
                        # Note: this only works as options are copied in each advanced technique function
                        options_before = options

                        # Use for logging
                        options_temp = options

                        for technique_cleanup in included_techniques:

                            # TODO Centralise and reuse this code
                            if show_logs: print(format_bold(f"Applying {technique_cleanup}"))

                            time_start = time.time()
                            number_removed_values, hits, options, details = apply_technique(technique_cleanup, options, chars, solved_instance, show_logs=show_logs)
                            log_cleanup_steps.append((technique_latest, technique_cleanup, counter_subprocess, options_before, details))
                            time_end = time.time()
                            times[technique_cleanup] += time_end - time_start
                            counts[technique_cleanup] += 1

                            if len(hits) > 0:
                                assert len(hits) == 1

                                # Count the most difficult technique, which is the technique defined in the outer loop
                                counts_techniques[technique_latest] += len(hits)
                                new_idxs.extend([t[0] for t in hits])

                                for hit in hits:
                                    assert 2 <= len(hit) <= 4
                                    (i1, i2) = hit[0]
                                    char = hit[1]
                                    solved_instance[i1][i2] = char
                                    options[i1][i2] = set()
                                found_hit = True
                                # TODO Refactor code
                                log_new_values_found = hits
                                log_technique_used = technique_latest
                                log_is_cleanup_issue = technique_cleanup != technique_latest
                                break

                            # Logging
                            if options_temp != options:
                                if show_logs:
                                    print(f"Options were removed when applying {technique_cleanup}")
                            options_temp = options

                        if found_hit:
                            if show_logs:
                                print(f"Found a new value during this iteration with {technique_cleanup}!")
                            break
                        elif options != options_before:
                            if show_logs:
                                print("Did not find a new value during this iteration, but at least one options was removed, try another iteration..")
                            continue
                        else:
                            if show_logs:
                                print("No new value found and no option removed, try to add the next advanced technique")
                            break

                    if found_hit:
                        could_use_technique = True
                        break

            # if could_use_technique:
            #     print(solved_instance)
            #     raise Exception("This is the new logic, which should be reached at least sometimes")

        # Step 4: Magic
        if not could_use_technique:
            if show_logs:
                print(solved_instance)
                print("Could not use any technique to find new value, investigate..")
            if include_magic_technique:
                idxs = [(i1, i2) for i1 in range(size) for i2 in range(size) if solved_instance[i1][i2] == EMPTY_CHAR]
                idx = random.choice(idxs)
                i1, i2 = idx
                solved_instance[i1][i2] = magic_solution[i1][i2]
                counts_techniques["magic"] += 1
                new_idxs.extend([idx])
            else:
                break

        if show_logs:
            output = format_solution(solved_instance, highlight_idxs=new_idxs)
            print(output)
            print(dict(sorted(counts_techniques.items(), key=lambda item: TECHNIQUES.index(item[0]))))
            print(" ")

        # Update logs
        log_solved_instance_after = copy_instance(solved_instance)
        logs["steps"].append(
            # TODONE Hits should contain more information: Technique used and extra information on how the value was
            #  found; For singles we already include this as the third tuple element, we can use this structure for
            #  all techniques
            # TODO Instead of only including the technique that was used to find the final value, we should create a
            #  list with technique applications which removed at least one option; Based on this we should create logic
            #  which determines the sequence of applications in order to find the new value
            (
                log_solved_instance_before, log_solved_instance_after,
                log_new_values_found, log_technique_used,
                log_is_cleanup_issue, log_cleanup_steps,
            )
        )

        # Intermediate verifications
        if debug_mode:

            # Partially verify
            def _verify_dimension(dim, dim_name=None):
                for char in chars:
                    try:
                        assert dim.count(char) <= 1, f"Something wrong with the partial solution: {dim_name} has duplicate char '{char}'.."
                    except ValueError:
                        pass

            try:
                for idx, row in solved_instance.get_rows():
                    _verify_dimension(row, dim_name=f"row-{idx + 1}")
                for idx, col in solved_instance.get_cols():
                    _verify_dimension(col, dim_name=f"col-{idx + 1}")
                for idxs, box in solved_instance.get_boxs():
                    _verify_dimension(box, dim_name=f"box-{tuple(map(lambda x: x + 1, idxs))}")
            except Exception as e:
                print(solved_instance)
                raise e

            # Check that the partial solution still matches the true solution, if it is given
            if magic_solution is not None:
                for i1 in range(size):
                    for i2 in range(size):
                        filled_value = solved_instance[i1][i2]
                        actual_value = magic_solution[i1][i2]
                        assert filled_value == EMPTY_CHAR or filled_value == actual_value, \
                            f"Filled value not correct! ('{filled_value}' instead of '{actual_value}' at {(i1 + 1, i2 + 1)})"

        # Update while-condition
        is_filled = count_empty_cells(solved_instance) == 0

    if debug_mode:
        if is_filled:
            try:
                assert sum(counts_techniques.values()) == count_empty_cells(instance)
            except AssertionError as e:
                print()
                print(instance)
                print(counts_techniques)
                raise e

    counts_techniques = dict(sorted(counts_techniques.items(), key=lambda item: TECHNIQUES.index(item[0])))
    if show_logs:
        print("Techniques used:", counts_techniques)

    # TODO Apply function with weights

    # Note: The instance does not necessarily be solved -- In some places we check whether the returned solution is
    #  completely filled in; Though it might be better to raise an exception in case this is the case
    # assert count_empty_cells(solved_instance) == 0, "Instance could not be solved with specified human techniques!"

    # I should have added logs here from the start, when adding later all calls to the function have to be updated;
    # Now I might be lucky that step can be reused, as it does not seem to be used anywhere; To avoid having to rewrite
    # the return arguments later when more information is added to logs, make it a dictionary

    return solved_instance, (counts_techniques, logs)


def determine_is_solvable_using_human_techniques(instance, solution):
    assert isinstance(instance, Instance)
    assert isinstance(solution, Instance)
    # Note: By default, the magic technique is not considered, even though it is in the list of techniques
    # TODO Remove it from this list
    solved_instance, _ = solve_using_human_techniques(instance, use_techniques=TECHNIQUES)
    return solved_instance == solution


def format_solution(solution, highlight_idxs):
    output = ""
    for i1, row in enumerate(solution):
        output += '['
        for i2, e in enumerate(row):
            text = f"'{e}'"
            # if (i1, i2) in highlight_idxs:
            #     text = colorama.Fore.RED + text + colorama.Fore.RESET
            output += text
            if i2 < len(row) - 1:
                output += ", "
        output += ']'
        output += '\n'
    return output


def format_bold(text):
    return "\033[1m" + text + "\033[0m"


def _remove_duplicate_hits(hits):

    # print("Before removing duplicates:", hits)

    _hits = [hit[:2] for hit in hits]

    # Remove duplicates (convert to dict and then convert back to list)
    hits = sorted(
        # TODO Remove the if/else exception when all techniques are implemented
        [(*k, *v) for k, v in {hit[:2]: hit[2:] if len(hit) > 2 else [None] for hit in hits}.items()],
        key=lambda t: t[:2]
    )

    # This only worked when no details were added as the third element
    # hits = sorted(set(hits))

    # print("After removing duplicates:", hits)

    assert all(hit[:2] in _hits for hit in hits)

    return hits


# Temporarily created a function to introduce new structure, but we might keep it as the logic is now scattered in
#  3 places and centralising it makes sense -> we definitely want to keep it, as the logic is now quite extensive, and
#  we might want to add more shared logic to it
def apply_technique(name, options, chars, solved_instance, show_logs):

    fnc = fncs[name]

    # TODO For now these only require the instance, we might want to give this to all functions
    # TODO We might also want to remove chars as argument, which should be included in options
    techniques_requiring_instance = [
        "boxed-doubles", "boxed-triplets", "boxed-quads",
        "boxed-rays",
    ]

    if name in techniques_requiring_instance:
        fnc = partial(fnc, instance=solved_instance)

    # Apply technique
    details = fnc(options, chars, show_logs=show_logs)

    # TODO With this logic we could also potentially determine the min number of applications needed to find a new
    #  value

    # Options are NOT modified anymore inside the function when applying the technique, but instead details are
    #  gathered and options are removed afterwards;
    #  TODO We could ensure this by making it immutable

    # Remove options
    # Note: Removing options is already logged inside the technique, where it should be done
    if show_logs:
        print(" >> Removing options")
    # Before removing options afterwards, copy the instance (note that inside technique functions options are not
    #  updated anymore)
    options = copy_options(options)
    import itertools
    number_options_before = sum(map(len, itertools.chain(*options)))
    num_removed_options = 0
    # TODO Find a better way to do this -- For now we have to indicate which options were actually removed by each
    #  application in order to trace back how the new value was found
    details_updated = []
    for name_application, application_details, removed_chars in details:
        # if show_logs:
        #     print(f"Application {name_application} removed {len(removed_chars)} options")
        removed_chars_updated = []
        for (i1, i2), char in removed_chars:
            # Note: Multiple applications might remove the same option, so we have to do a check
            try:
                options[i1][i2].remove(char)
                if show_logs:
                    print(f" Remove '{char}' at {(i1 + 1, i2 + 1)}")
                num_removed_options += 1
                removed_chars_updated.append(((i1, i2), char))
            except KeyError:
                pass
        # Only keep applications which removed at least one option, to avoid clogging the logs
        if len(removed_chars_updated) > 0:
            # TODO This is a temporary ugly hack until we have defined a better structure
            #  -> Note that this does not have an impact on writer/messages loops as the application name is not used
            details_updated.append(((name, name_application), application_details, removed_chars_updated))
    # if show_logs:
    #     print(f"Total number of options removed: {num_removed_options}")
    number_options_after = sum(map(len, itertools.chain(*options)))
    number_options_removed = number_options_before - number_options_after
    assert number_options_removed >= 0

    # Identify new values
    if show_logs:
        print(" >> Identifying new values")
    new_values = identify_new_values(options, chars, show_logs=show_logs)

    # Exception for techniques based on chaining, which are expected to find a new value in the conflicting cell
    # TODO Remove later when the techniques allow for removing options only
    if name in ["remote-pairs", "ab-chains"]:
        new_values = [
            (idx, char, "cell")
            for (idx, char, dims) in new_values
            if "cell" in dims
        ]

    # Add details to be consistent with current code
    # TODO Restructure
    new_values = [(*new_value, details_updated) for new_value in new_values]

    if show_logs:
        for (i1, i2), char, _dims, _details in new_values:
            # TODO We could make this more detailed by trying to find new values after removing an option, and show
            #  the application name instead
            print(f" -> After applying {name}, found new value {char} at {(i1 + 1, i2 + 1)}")

    number_removed_values = num_removed_options
    hits = new_values

    # Added more shared logic
    hits = _remove_duplicate_hits(hits)

    if show_logs:
        print(f" Removed {number_removed_values} options with {name}, and found {len(hits)} new values:")
        for t in hits:
            print(" ", (tuple(map(lambda x: x + 1, t[0])), *t[1:3]))

    # Make sure to only include one value
    # TODO Select one randomly would be better
    hits = hits[:1]

    # TODO Remove details from hits
    # TODO Include processing hits in the centralised function

    return number_removed_values, hits, options, details_updated


def parse_cleanup_logs(logs):
    logs_parsed = []

    steps = logs["steps"]
    for idx, step in enumerate(steps):
        steps_cleanup = step[-1]

        step_parsed = []
        for step_cleanup in steps_cleanup:
            _, _, iteration, options_before_application, details = step_cleanup
            if len(details) > 0:
                for ((name_technique, name_application), _, removed_chars) in details:
                    _removed_chars = [(tuple(map(lambda x: x + 1, idx)), char) for idx, char in removed_chars]
                    if len(_removed_chars) > 0:
                        step_parsed.append((name_technique, name_application, removed_chars, options_before_application))

        logs_parsed.append(step_parsed)

    return logs_parsed
