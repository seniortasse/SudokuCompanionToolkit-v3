
from collections import defaultdict, Counter
import itertools
import operator

from generator.techniques.advanced import get_idxs_in_dimensions


def find_remote_pairs(options, chars, show_logs=False):

    # Steps:
    #  1 For each cell, determine whether it contains a pair in any dimension
    #  2 For all other cells in a candidate cell's dimensions, determine whether it contains pairs which have 1 value in
    #    common
    #  3 If there are more than 2 such values, we have the start/end of a chain (we assume we already applied the
    #    "doubles" technique, and therefore the found pairs cannot be a standard 1-dim doubles)
    #  4 Determine whether there is a valid chain between each possible endpoints

    # TODO Update: Instead of requiring the starting point to contain a pair of which one is excluded by the chain
    #  (restricted definition), a more general definition is that the chain excludes the pair in the head/tail of the
    #  chain from all cells sharing a dimension with both endpoints

    details_with_lengths = []

    for (i1, i2) in itertools.product(range(options.size), repeat=2):
        options_for_cell = options[i1][i2]
        if len(options_for_cell) == 2:
            if show_logs:
                print(f"Candidate cell for remote pair: {(i1 + 1, i2 + 1)}: {options_for_cell}")

            # Define idxs to check
            idxs = get_idxs_in_dimensions(options, i1, i2)

            # Identify similar pairs in row
            potential_pairs_and_endpoints = defaultdict(list)
            for (_i1, _i2) in idxs:
                _options_for_cell = options[_i1][_i2]
                options_in_common = options_for_cell.intersection(_options_for_cell)
                if len(_options_for_cell) == 2 and len(options_in_common) == 1:
                    # print(f"Found potential endpoint at {(_i1, _i2)} with pair {_options_for_cell}")
                    potential_pairs_and_endpoints[tuple(sorted(_options_for_cell))].append((_i1, _i2))

            for potential_pair, potential_endpoints in potential_pairs_and_endpoints.items():
                if len(potential_endpoints) >= 2:
                    if show_logs:
                        print(f"Found potential pair {potential_pair} with at least 2 possible endpoints: {[tuple(e + 1 for e in idx) for idx in potential_endpoints]}")

                    for endpoint_start, endpoint_end in itertools.combinations(potential_endpoints, 2):
                        if show_logs:
                            print(f"Try starting at {tuple(map(lambda x: x + 1, endpoint_start))} and ending at {tuple(map(lambda x: x + 1, endpoint_end))}")

                        valid_chains = search_valid_chains(
                            options, potential_pair, endpoint_start, endpoint_end, show_logs
                        )

                        # Determine whether a valid chain was found
                        if len(valid_chains) > 0:
                            removed_char = options_for_cell.intersection(potential_pair)
                            assert len(removed_char) == 1
                            removed_char = removed_char.pop()
                            only_possible_char = options_for_cell.difference(potential_pair)
                            assert len(only_possible_char) == 1
                            only_possible_char = only_possible_char.pop()

                            if show_logs:
                                print(f"Found {len(valid_chains)} valid chain(s) for pair {potential_pair}, "
                                      f"which makes '{only_possible_char}' the only option at {(i1 + 1, i2 + 1)}")
                                print(" Chain lengths:", dict(sorted(Counter(map(len, valid_chains)).items(), key=lambda item: item[0])))

                            # We don't have to update the options as they are not used to find the new value;
                            #  However, we should restructure this and find the new value based on options as with
                            #  all other advanced techniques
                            # options[i1][i2] = only_possible_char

                            # Add details
                            removed_chars = [((i1, i2), removed_char)]

                            idx_start = (i1, i2)
                            valid_chain = sorted(valid_chains, key=len)[0]  # Select the shortest chain
                            options_for_idxs = {idx: options[idx[0]][idx[1]].copy() for idx in valid_chain}
                            options_for_idxs.update({idx_start: options[idx_start[0]][idx_start[1]]})  # Used to identify earlier applications
                            application_details = (idx_start, valid_chain, options_for_idxs)

                            # details = [(application_details, removed_chars)]

                            # Note: Temporarily add the length of the chain to easily select the shortest one later
                            # values.append(((i1, i2), only_possible_char, "cell", details, len(valid_chain)))

                            # Note: Finding the overall shortest chain is done in two steps:
                            #  - Select the shortest chain for each new value (above)
                            #  - Select the value with the overall shortest chain (end of function)
                            name_application = f"remote-pairs for {tuple(sorted(options_for_cell))} in {(i1 + 1, i2 + 1)} " \
                                               f"with chain {list(map(lambda idx: tuple(map(lambda x: x + 1, idx)), valid_chain))}"
                            details_with_lengths.append(((name_application, application_details, removed_chars), len(valid_chain)))

    # Finally, select the value found with the overall shortest chain
    # values = sorted(values, key=lambda value: value[-1])[:1]
    details = sorted(details_with_lengths, key=operator.itemgetter(1))[:1]

    if show_logs:
        print(f"Found {len(details_with_lengths)} new value(s) with remote-pairs, "
              f"using chains of lengths: {list(map(operator.itemgetter(1), details_with_lengths))}")
        if len(details) > 0:
            print(f"The overall shortest chain has length: {details[0][1]}")

    # Remove the temporarily added last element of the tuple
    # values = [value[:4] for value in values]
    # TODO Return all applications, but sort them, as the first will be selected
    details = [detail[0] for detail in details]

    # TODO We somehow have to make sure that the final value is found based on missing value in cell, which currently is
    #  the case as this dimension is checked first and the new values are ordered, but this is not explicitly guaranteed

    return details


# TODO We could and perhaps should use yield functionality here, as it might not be necessary to find all chains, if one
#  already satisfies the requirements
def search_valid_chains(options, potential_pair, endpoint_start, endpoint_end, show_logs):

    valid_chains = []

    # Recursion
    def follow_chain(chain):
        current_point = chain[-1]

        idxs_in_dimensions = get_idxs_in_dimensions(options, *current_point)

        potential_next_points = [
            idx for idx in idxs_in_dimensions
            if len(options[idx[0]][idx[1]].symmetric_difference(potential_pair)) == 0
               and idx not in chain
        ]

        if show_logs:
            print("Potential next points:", [tuple(e + 1 for e in idx) for idx in potential_next_points])

        for potential_next_point in potential_next_points:
            chain_copy = chain.copy()
            chain_copy.append(potential_next_point)

            if potential_next_point == endpoint_end and len(chain_copy) % 2 == 0:
                if show_logs:
                    print("Found valid chain!", [tuple(e + 1 for e in idx) for idx in chain_copy])
                valid_chains.append(chain_copy)
            else:
                # Keep following the chain
                follow_chain(chain_copy)

    # Initialise chain
    chain = [endpoint_start]

    # Find chains
    follow_chain(chain)

    return valid_chains
