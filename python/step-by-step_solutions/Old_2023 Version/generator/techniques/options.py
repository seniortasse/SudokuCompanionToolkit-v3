
import itertools
from operator import itemgetter

from generator.model import EMPTY_CHAR, Instance, DIMENSIONS


# This already seems to include advanced logic, this is not "singles", but can be doubles/triplets as well
# TODO Integrate with fits() in model
def determine_options(instance, i1, i2, check_row=True, check_col=True, check_box=True):
    assert instance[i1][i2] == EMPTY_CHAR
    relevant_chars = set()
    if check_row:
        row = instance.get_values("row", i1)
        relevant_chars.update(row)
    if check_col:
        col = instance.get_values("col", i2)
        relevant_chars.update(col)
    if check_box:
        box = instance.get_values("box", instance.get_idx_box(i1, i2))
        relevant_chars.update(box)
    options = instance.chars.difference(relevant_chars)
    return options


# This should only be called once when we are starting to apply more advanced techniques
def determine_options_per_cell(instance):
    collection_options = [[] for _ in range(instance.size)]
    for (i1, i2) in itertools.product(range(instance.size), repeat=2):
        if instance[i1][i2] == EMPTY_CHAR:
            options_for_cell = determine_options(instance, i1, i2)
        else:
            options_for_cell = set()
        collection_options[i1].append(options_for_cell)
    return Instance(collection_options, is_chars=False, preprocessed_dims=instance.dims)


def copy_options(options):
    copy = [
        [
            e.copy() if e else e
            for e in row
        ]
        for row in options
    ]
    # Note: We only use an instance for options to make printing more readable; We do not use any dimensions, as these
    #  are read from the actual instance, so we do not need to create a neat copy here
    #  -> We do actually need to dimensions
    return Instance(copy, is_chars=False, preprocessed_dims=options.dims)


# TODO Combine this with singles techniques
def identify_new_values(options, chars, show_logs=False):
    values = []

    # Single occurrence in cell
    for (i1, i2) in itertools.product(range(options.size), repeat=2):
        if len(options[i1][i2]) == 1:
            char = options[i1][i2].copy().pop()
            values.append(((i1, i2), char, "cell"))
            if show_logs:
                print(f"Found new char based on only possible occurrence in cell: {char} at {(i1 + 1, i2 + 1)}")

    # Single occurrence in dim
    for dim in DIMENSIONS:
        for idx_dim, idxs_for_dim in options.idxs_for_dims[dim].items():
            for char in chars:
                idxs_possible = [
                    (i1, i2)
                    for (i1, i2) in idxs_for_dim
                    if char in options[i1][i2]
                ]
                if len(idxs_possible) == 1:
                    (i1, i2) = idxs_possible[0]
                    values.append(((i1, i2), char, dim))
                    if show_logs:
                        print(f"Found new char based on only possible occurrence in {dim} {idx_dim + 1}: '{char}' at {(i1 + 1, i2 + 1)}")

    # Group duplicate new values
    if len(values) > 1:
        values = _group_values(values)

    return values


def _group_values(values):
    # Similar as for the logic of finding singles based on values (here on options), we want to know how the new value
    #  was found (only here we only distinguish between only occurrence in cell or dim), this will be used in the logs

    grouped_values = []
    # IMPORTANT: groupby() only groups similar items when they are consecutive, if they are scattered through the list
    #  they will each be in a separate group (this doesn't make a lot of sense..), therefore we have to sort the list
    #  first based on the key
    sort_key = itemgetter(0)
    values = sorted(values, key=sort_key)
    grouper = itertools.groupby(values, sort_key)  # Group by idx
    for key, group in grouper:
        group = list(group)  # Unpack
        assert len(set(value[1] for value in group)) == 1, f"Identified new values for {key} inconsistent: {group}"
        grouped_dims = ' & '.join(value[2] for value in group)
        grouped_value = (key, group[0][1], grouped_dims)
        grouped_values.append(grouped_value)

    # Temporary health check
    grouped_values_old = list(set([value[:2] for value in values]))
    assert len(grouped_values) == len(grouped_values_old)
    assert sorted(t[:2] for t in grouped_values) == sorted(t[:2] for t in grouped_values_old)

    return grouped_values

