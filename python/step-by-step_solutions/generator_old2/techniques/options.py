
import itertools
from operator import itemgetter

from generator.model import EMPTY_CHAR, Instance, get_box


# This already seems to include advanced logic, this is not "singles", but can be doubles/triplets as well
def determine_options(instance, i1, i2, check_row=True, check_col=True, check_box=True):
    assert instance[i1][i2] == EMPTY_CHAR
    relevant_chars = set()
    if check_row:
        row = instance[i1]
        relevant_chars.update(row)
    if check_col:
        col = [row[i2] for row in instance]
        relevant_chars.update(col)
    if check_box:
        box = get_box(instance, i1, i2)
        relevant_chars.update(box)
    options = instance.chars.difference(relevant_chars)
    return options


# This should only be called once when we are starting to apply more advanced techniques
def determine_options_per_cell(instance):
    collection_options = []
    size = instance.size
    for i1 in range(size):
        options = []
        for i2 in range(size):
            if instance[i1][i2] == EMPTY_CHAR:
                options_for_cell = determine_options(instance, i1, i2)
            else:
                options_for_cell = set()
            options.append(options_for_cell)
        collection_options.append(options)
    return Instance(collection_options, layout=instance.layout, is_chars=False)


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
    return Instance(copy, layout=options.layout, is_chars=False)


def identify_new_values(options, chars, show_logs=False):
    values = []
    box_height, box_width, size = options.box_height, options.box_width, options.size
    for i1 in range(size):
        for i2 in range(size):
            if len(options[i1][i2]) == 1:
                char = options[i1][i2].copy().pop()
                values.append(((i1, i2), char, "cell"))
                if show_logs:
                    print(f"Found new char based on only possible occurrence in cell: {char} at {(i1 + 1, i2 + 1)}")
    for i1 in range(size):
        for char in chars:
            possible_idxs = []
            for i2 in range(size):
                if char in options[i1][i2]:
                    possible_idxs.append(i2)
            if len(possible_idxs) == 1:
                i2 = possible_idxs[0]
                values.append(((i1, i2), char, "row"))
                if show_logs:
                    print(f"Found new char based on only possible occurrence in row {i1 + 1}: {char} at {(i1 + 1, i2 + 1)}")
    for i2 in range(size):
        for char in chars:
            possible_idxs = []
            for i1 in range(size):
                if char in options[i1][i2]:
                    possible_idxs.append(i1)
            if len(possible_idxs) == 1:
                i1 = possible_idxs[0]
                values.append(((i1, i2), char, "col"))
                if show_logs:
                    print(f"Found new char based on only possible occurrence in col {i2 + 1}: {char} at {(i1 + 1, i2 + 1)}")
    for b1 in range(size // box_height):
        for b2 in range(size // box_width):
            for char in chars:
                possible_idxs = []
                for _i1 in range(box_height):
                    for _i2 in range(box_width):
                        i1 = box_height * b1 + _i1
                        i2 = box_width * b2 + _i2
                        if char in options[i1][i2]:
                            possible_idxs.append((i1, i2))
                if len(possible_idxs) == 1:
                    i1, i2 = possible_idxs[0]
                    values.append(((i1, i2), char, "box"))
                    if show_logs:
                        print(f"Found new char based on only possible occurrence in box {(b1 + 1, b2 + 1)}: {char} at {(i1 + 1, i2 + 1)}")

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

