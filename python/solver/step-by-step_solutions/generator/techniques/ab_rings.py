
from collections import Counter
import itertools


def find_ab_rings(options, chars, show_logs=False):

    # Another chaining technique (besides remote-pairs and ab-chains), which requires a combination of 4 cells in a
    #  square pattern (ring), where each cell contains a pair, shares one value with the neighboring cells in the chain,
    #  and all cells in the chain combined contain a combination of 4 options (a quad);

    # Steps:
    #  - Identify cells with a pair
    #  - Determine all possible ring formations
    #  - Check whether the ring contains a combination of 4 options
    #  - Apply the removal pattern, which is removing the shared value for adjacent cells in the chain from the other
    #    cells in the dimension they share (row/col)

    size = options.size

    details = []

    # Preprocessing data structure
    # idxs_with_pairs = [(i1, i2) for (i1, i2) in itertools.product(range(size), repeat=2) if len(options[i1][i2]) == 2]
    idxs_with_pairs = {i1: [i2 for i2 in range(size) if len(options[i1][i2]) == 2] for i1 in range(size)}

    # Identify rings by looping over rows, checking for all combinations of two cells containing a pair whether there
    #  is another row which contain pairs in the same cols
    idxs_rings = []
    for i1 in range(size):
        # idxs_cols = [i2 for i2 in range(size) if (i1, i2) in idxs_with_pairs]
        idxs_cols = idxs_with_pairs[i1]
        combs_idxs = itertools.combinations(idxs_cols, 2)
        for comb_idxs in combs_idxs:
            for _i1 in range(size):
                if _i1 > i1:  # We don't have to find the ring twice
                    _idxs_cols = idxs_with_pairs[_i1]
                    if all(idx in _idxs_cols for idx in comb_idxs):
                        # Found a valid ring!
                        idxs_rings.append(((i1, _i1), comb_idxs))

    # Postprocess idxs
    idxs_rings = [
        [
            (i1, i2) for (i1, i2) in itertools.product(idxs_ring[0], idxs_ring[1])
            # [options[i1][i2] for (i1, i2) in itertools.product(idxs_ring[0], idxs_ring[1])],
        ]
        for idxs_ring in idxs_rings
    ]

    # Reverse the bottom to make it a chain
    idxs_rings = [
        idxs_ring[:2] + idxs_ring[2:][::-1]
        for idxs_ring in idxs_rings
    ]

    if show_logs:
        print("Rings found:")
        for idxs_ring in idxs_rings:
            print("", list(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_ring)))  # , "-", _options)

    # Adjacent cells in the ring should have only 1 value in common
    # idxs_rings_filter_1 = [
    #     idxs_ring
    #     for idxs_ring in idxs_rings
    #     if all(
    #         len(set.intersection(
    #             options[idxs_ring[i - 1][0]][idxs_ring[i - 1][1]],
    #             options[idxs_ring[i][0]][idxs_ring[i][1]],
    #         )) == 1
    #         for i in range(4)
    #     )
    # ]

    # print([
    #     (idxs_ring, [
    #         set.intersection(
    #             options[idxs_ring[i - 1][0]][idxs_ring[i - 1][1]],
    #             options[idxs_ring[i][0]][idxs_ring[i][1]],
    #         )
    #         for i in range(4)
    #     ])
    #     for idxs_ring in idxs_rings
    # ])

    # if show_logs:
    #     print("Rings containing only adjacent cells with a single shared option:")
    #     for idxs_ring in idxs_rings_filter_1:
    #         print("", tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_ring)))

    # for idxs_ring in idxs_rings:
    #     is_valid = True
    #     for i1 in idxs_rings[0]:
    #         is_valid = is_valid and len(set.intersection(*(options[i1][i2] for i2 in idxs_rings[1]))) == 1
    #     for i2 in idxs_rings[1]:
    #         is_valid = is_valid and len(set.intersection(*(options[i1][i2] for i1 in idxs_rings[0]))) == 1
    #     if is_valid:
    #         idxs_rings_filter_1.append(idxs_ring)

    # Only rings containing a combination of 4 values are relevant
    # idxs_rings_filter_2 = [
    #     idxs_ring
    #     for idxs_ring in idxs_rings
    #     if len(set.union(*( options[i1][i2] for (i1, i2) in idxs_ring))) == 4
    # ]

    # if show_logs:
    #     print("Rings containing a quad:")
    #     for idxs_ring in idxs_rings_filter_2:
    #         print("", tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_ring)))

    # idxs_rings_relevant = [
    #     idxs_ring
    #     for idxs_ring in idxs_rings
    #     if idxs_ring in idxs_rings_filter_1 and idxs_ring in idxs_rings_filter_2
    # ]

    # Updated condition for a valid ring: Should be an ab-chain with a combination of 4 options (alternative naming for
    #  this technique could be bent-quads) - this can be determined by checking whether all adjacent cells have 1 option
    #  in common and each option occurs twice in the ring
    idxs_rings_relevant = []
    for idxs_ring in idxs_rings:

        # Condition 1: Adjacent cells with 1 shared option
        is_condition_1_valid = all(
            len(set.intersection(
                options[idxs_ring[i - 1][0]][idxs_ring[i - 1][1]],
                options[idxs_ring[i][0]][idxs_ring[i][1]],
            )) == 1
            for i in range(4)
        )

        # print("Condition 1")
        # print([
        #     set.intersection(
        #         options[idxs_ring[i - 1][0]][idxs_ring[i - 1][1]],
        #         options[idxs_ring[i][0]][idxs_ring[i][1]],
        #     )
        #     for i in range(4)
        # ])

        # Condition 2: All options occur twice
        all_options = list(itertools.chain.from_iterable(
            options[idxs_ring[i][0]][idxs_ring[i][1]] for i in range(4)
        ))
        is_condition_2_valid = all(count == 2 for _, count in Counter(all_options).items())

        # print("Condition 2")
        # print(Counter(all_options))

        if is_condition_1_valid and is_condition_2_valid:
            assert len(set(all_options)) == 4
            idxs_rings_relevant.append(idxs_ring)

    if show_logs:
        print("Rings relevant:")
        for idxs_ring in idxs_rings_relevant:
            print("", tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_ring)))

    # Remove options for valid rings
    for idxs_ring in idxs_rings_relevant:
        combined_options = set.union(*(
            options[i1][i2]
            for (i1, i2) in idxs_ring
        ))
        # combined_options = _options
        assert len(combined_options) == 4

        name_application = f"ab-rings in {tuple(map(lambda idx: tuple(map(lambda x: x + 1, idx)), idxs_ring))} with quad {tuple(sorted(combined_options))}"
        if show_logs:
            print(f"Found a {name_application}")

        idxs_rows = tuple(sorted(set(idx[0] for idx in idxs_ring)))
        idxs_cols = tuple(sorted(set(idx[1] for idx in idxs_ring)))
        assert len(idxs_rows) == 2
        assert len(idxs_cols) == 2

        removed_chars = []
        for i1 in idxs_rows:
            shared_options = set.intersection(*(options[i1][i2] for i2 in idxs_cols))
            assert len(shared_options) == 1
            shared_option = shared_options.pop()
            if show_logs:
                print(f"Removing shared option {shared_option} from row {i1} outside of cols {tuple(map(lambda x: x + 1, idxs_cols))}")
            for i2 in range(size):
                if i2 not in idxs_cols:
                    if shared_option in options[i1][i2]:
                        if show_logs:
                            print(f"Remove '{shared_option}' from {(i1 + 1, i2 + 1)}")
                        removed_chars.append(((i1, i2), shared_option))
        for i2 in idxs_cols:
            shared_options = set.intersection(*(options[i1][i2] for i1 in idxs_rows))
            assert len(shared_options) == 1
            shared_option = shared_options.pop()
            if show_logs:
                print(f"Removing shared option {shared_option} from col {i2} outside of rows {tuple(map(lambda x: x + 1, idxs_rows))}")
            for i1 in range(size):
                if i1 not in idxs_rows:
                    if shared_option in options[i1][i2]:
                        if show_logs:
                            print(f"Remove '{shared_option}' from {(i1 + 1, i2 + 1)}")
                        removed_chars.append(((i1, i2), shared_option))

        options_for_idxs = {idx: options[idx[0]][idx[1]] for idx in idxs_ring}
        application = (idxs_ring, options_for_idxs, combined_options)
        details.append((name_application, application, removed_chars))

    return details
