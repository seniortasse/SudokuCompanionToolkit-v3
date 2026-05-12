
import itertools
import operator

from generator.techniques.advanced import get_idxs_in_dimensions


# TODO We can improve ab-chains in two ways:
#  - do an exhaustive search to find the shortest chain,
#  - does replacing dfs with bfs work here?


# Used to abort the recursion, not sure if there is cleaner implementation
class ValueFoundException(Exception):
    pass


def find_ab_chains(options, chars, show_logs=False):

    # Similar to the "remote pairs" technique, which requires a chain of pairs all having a single value in common;
    # Instead, this technique requires that for either value (A or B) in the starting cell, a chain of pairs can be
    #  found which invalidates the choice (A or B), and therefore the other value has to be filled in; This time, the
    #  chain is not continued by cells having a pair sharing some value, but by cells having a pair which contains the
    #  last value of the chain (so that the other value of the pair is enforced) - note that this could lead to chains
    #  of pairs which do not necessarily have some value in common

    # Steps:
    #  1 Iterate through all cells containing a pair, use this cell as the starting point
    #  2 Iterate through both options of the cell (A and B)
    #  3 For all other cells in the dimension of the last cell of the chain (initially the starting cell), determine
    #    whether there is a pair containing the last value of the chain, in which case the alternative value is
    #    enforced
    #  4 This is done recursively, until there is no continuation of the chain anymore
    #    - Note that we shouldn't stop when a valid chain is found, as the chain might continue, which might otherwise
    #      not be found
    #    - Make sure we are not making loops by checking that the new cell is not already contained in the chain
    #  5 When we arrive at a cell in the same dimension of the starting cell, check whether the enforced value of the
    #    chain tail is in conflict with the chosen value (A or B) of the starting point; If so, the only option is the
    #    other value (B of started with A, A if started with B)

    # TODO Update: Instead of requiring the starting point to contain a pair of which one is excluded by the chain
    #  (restricted definition), a more general definition is that the chain excludes the shared value of the head/tail
    #  of the chain from all cells sharing a dimension with both endpoints (similar to improvement for remote-pairs)

    # Implementation note:
    #  - We explicitly abort the search when one new value is found, as we do not need to know ALL values at the same
    #    time - when one value is found we might find the next value with an easier technique (note that this is
    #    currently inconsistent with the other advanced techniques, which search exhaustively for all new values)

    # Note: This makes the code very very fast, as we abort immediately if a value is found
    #  However for readable user logs we want to find the shortest chain, and therefore have to make the search a bit
    #  smarter
    ABORT_ON_HIT = False
    FIND_SHORTEST = True

    box_height, box_width, size = options.box_height, options.box_width, options.size
    dims = box_height, box_width, size

    # Note: We do not update the options when applying this technique, so do not have to make a copy nor to worry about
    #  the updated options issue

    # values = []

    # details = []
    details_with_lengths = []

    # 1
    cell_idxs_with_pairs = []
    for i1, i2 in itertools.product(range(size), repeat=2):
        options_for_cell = options[i1][i2]
        if len(options_for_cell) == 2:
            cell_idxs_with_pairs.append((i1, i2))

    if show_logs:
        print("Number of cells with a pair:", len(cell_idxs_with_pairs))
        print(cell_idxs_with_pairs)

    # We want to find the shortest chain
    min_len_conflicting_chain = float("inf")

    # 2
    for idx_start in cell_idxs_with_pairs:
        # Already found a value, explicitly aborting
        # Note: We add an entry to this list every time a new value is found
        if ABORT_ON_HIT and len(details_with_lengths) > 0:
            break
        if show_logs:
            print(f"Use {idx_start} as starting point")
        options_for_cell = options[idx_start[0]][idx_start[1]]
        for starting_value in options_for_cell:
            if show_logs:
                print(f" Try filling in {starting_value}")

            # conflict_found = False

            # For logging purposes
            # count_chains = 0
            # chains = []
            # chain_conflict = None
            chains_conflicting = []

            def follow_chain(chain_idxs, previous_value):
                nonlocal chains_conflicting, min_len_conflicting_chain

                # Note: We need the full chain to determine whether a new cell was already visited
                previous_idx = chain_idxs[-1]

                # Find all other cells in the dimension
                idxs_in_dimensions = get_idxs_in_dimensions(dims, *previous_idx)

                # End-of-chain condition (note: we still continue the search)
                if idx_start in idxs_in_dimensions:
                    # if show_logs:
                    #     nonlocal count_chains
                    #     count_chains += 1
                        # chains.append(chain_idxs)
                    if previous_value == starting_value:
                        # conflict_found = True
                        len_chain = len(chain_idxs)
                        if show_logs:
                            print(f"Found a conflicting AB-chain: ({len_chain})", chain_idxs)
                        # chain_conflict = chain_idxs
                        chains_conflicting.append(chain_idxs)
                        # Keep track of the overall shortest chain
                        min_len_conflicting_chain = min(min_len_conflicting_chain, len_chain)
                        # Extra stop condition
                        if ABORT_ON_HIT:
                            raise ValueFoundException()

                for idx_in_dimension in idxs_in_dimensions:

                    # Smart search for finding the shortest chain: Only continue the search if the current chain
                    #  is shorter than the currently found shortest chain
                    # Note: Since we are doing a dfs, we have to do this check at the start of the recursive function or
                    #  inside this loop, not before it
                    # if chain_conflict is not None and len(chain_idxs) == len(chain_conflict):
                    if FIND_SHORTEST and len(chain_idxs) == min_len_conflicting_chain - 1:
                        # print(f"Not following any further as the chain cannot be shorter than the shortest chain "
                        #       f"({min_len_conflicting_chain})")
                        return

                    # Avoid making loops
                    if idx_in_dimension in chain_idxs:
                        continue

                    # 3
                    # Only follow the chain along cells with a pair containing the last value (and therefore the other
                    #  value will be forced)
                    options_for_next_cell = options[idx_in_dimension[0]][idx_in_dimension[1]]
                    if len(options_for_next_cell) == 2 and previous_value in options_for_next_cell:
                        next_value = [value for value in options_for_next_cell if value != previous_value][0]

                        # if show_logs:
                        #     print(f"Follow chain {chain_idxs} to {idx_in_dimension}")

                        chain_idxs_copy = chain_idxs.copy()
                        chain_idxs_copy.append(idx_in_dimension)

                        follow_chain(chain_idxs_copy, next_value)

            # Initialise the chain
            chain_idxs = [idx_start]
            previous_value = starting_value

            # 4
            # Recursion
            try:
                follow_chain(chain_idxs, previous_value)
            except ValueFoundException:
                pass

            # 5
            # Check whether a conflict was found filling in the starting value and following the chains
            # conflict_found = chain_conflict is not None
            conflict_found = len(chains_conflicting) > 0
            if conflict_found:
                if FIND_SHORTEST:
                    chain_conflict = sorted(chains_conflicting, key=len)[0]
                else:
                    chain_conflict = chains_conflicting[0]

                enforced_value = [value for value in options_for_cell if value != starting_value][0]

                if show_logs:
                    print("Number of conflicting chains found:", len(chains_conflicting))
                    print("Found a new value with AB-chains!")
                    print(f" {enforced_value} at {idx_start} using chain {chain_conflict}")

                # Note: We could fill in the entire chain after this, but we only fill one value as to count the
                #  technique used only once

                # Add details; Note that we rely on the fact that an exception is thrown once a valid chain is found
                # Note: This technique removes one option in the first cell of the chain that leads to a conflict
                # Note: The starting cell always has two values (contains a double), the conflicting value (removed) and
                #  enforced value (filled in)
                removed_chars = [(idx_start, starting_value)]

                # TODO It would probably be cleaner to pass the entire initial options (and do this for all techniques)
                options_for_idxs = {idx: options[idx[0]][idx[1]].copy() for idx in chain_conflict}

                # Note: The application of ab-chains is somewhat different from other techniques - we do not identify
                #  new values through options (although we could do this), but fill in the new value manually
                # TODO It is probably cleaner to rewrite this to the structure we are using in all other techniques, ie
                #  removing an option and identifying the new value through the shared function
                conflicting_value = starting_value
                application_details = (chain_conflict, options_for_idxs, conflicting_value)

                # details = [(application_details, removed_options)]

                # values.append((idx_start, enforced_value, "cell", details))
                # Note: We only keep the latest found value, which leads to the shortest chain, relying on the fact that
                #  we only continue the search for the shortest chain
                # values = [(idx_start, enforced_value, "cell", details)]

                # Temporarily add the length of the chain to easily select the overall shortest chain
                # Note: We could just replace the details everytime we find a new value as we only continue the search
                #  for shorter chains, but this approach is more general
                name_application = f"ab-chains starting at {idx_start} " \
                                   f"with chain {list(map(lambda idx: tuple(map(lambda x: x + 1, idx)), chain_conflict))}"
                details_with_lengths.append(((name_application, application_details, removed_chars), len(chain_conflict)))

                # This breaks out of the inner loop when the starting value leads to a conflict, the other value does
                #  not need to be explored as it will be the filled in value and will not lead to a conflict
                break

                # Note: We break out of the outer loop at the start of the next iteration if ABORT_ON_HIT is set

            # if show_logs:
            #     print("Number of chains explored:", count_chains)
                # for chain in sorted(chains):
                #     print(chain)

    # Note: We do not remove any values from options, as this technique enables filling in the value immediately
    # TODO To be more precise, we actually do remove options from the value cell
    #  -> Now options will be removed outside of the function based on removed_chars

    # Finally select the shortest chain -> For now this is done in the logic

    # TODO Merge with logic in remote-pairs
    details = sorted(details_with_lengths, key=operator.itemgetter(1))[:1]

    if show_logs:
        print(f"Found {len(details_with_lengths)} new value(s) with ab-chains, "
              f"using chains of lengths: {list(map(operator.itemgetter(1), details_with_lengths))}")
        if len(details) > 0:
            print(f"The overall shortest chain has length: {details[0][1]}")

    details = [detail[0] for detail in details]

    return details
