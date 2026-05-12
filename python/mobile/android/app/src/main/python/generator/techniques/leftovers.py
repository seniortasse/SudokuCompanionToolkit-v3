
from collections import Counter
import itertools

from generator.model import EMPTY_CHAR


# A technique specifically applicable to instances with custom boxes layout, "leftovers" looks at a combination of one
#  or more adjacent rows or cols, and for all possible combinations of cells inside/outside this region possibly some
#  options can be removed; In particular, the options inside the region have to correspond to the options outside the
#  region, so only options inside the region will be kept outside the region and vice versa;


MAX_NUMBER_INSIDE = 9  # TODO Should this be based on the size?


cache_application_layouts = {}


# TODONE The valid applications are based on the layout, which remains the same for an instance, so we could do some
#  preprocessing to speed up the implementation for this technique - we would only have to determine the options
#  inside/outside, instead of preprocessing all idxs every time;
def _preprocess_layout(instance):
    """
    The possible applications need to be preprocessed only once, as they are based on the boxes layout which remains
     unmodified, and when finding valid applications only the options need to be checked;
    We store the preprocessed data in cache in this file, we could also store it in the instance but it is only needed
     when applying this technique, and we don't want to be bothered with adding any redundant error-prone logic;
    """

    size = instance.size

    application_layouts = []

    # Preprocessing
    regions = sorted((tuple(range(start, end + 1)) for start in range(size) for end in range(start, size)), key=len)
    assert len(regions) == size * (size + 1) // 2

    for dim in ["row", "col"]:
        for region in regions:

            idxs_for_region = _get_idxs_for_region(instance, dim, region)
            assert len(idxs_for_region) == len(region) * size and len(set(idxs_for_region)) == len(idxs_for_region)

            # Identify boxes overlapping the region and the number of cells inside the region
            box_counts = Counter(
                instance.get_idx_box(i1, i2)
                for (i1, i2) in idxs_for_region
            )
            assert all(v <= size for v in box_counts.values())

            # Only boxes which are not entirely in the region are relevant
            relevant_box_idxs = sorted((box_idx for box_idx, count in box_counts.items() if count < size))

            # Check all combinations of using a box for inside/outside, which is only valid if the number of cells
            #  inside/outside is the same
            box_idxs_combs = list(
                itertools.chain.from_iterable(
                    itertools.combinations(relevant_box_idxs, _size)
                    for _size in range(1, size + 1)
                )
            )

            for box_idxs_inside in box_idxs_combs:
                # TODO There might be a more efficient way of generating combinations
                box_idxs_outside = [box_idx for box_idx in relevant_box_idxs if box_idx not in box_idxs_inside]

                num_cells_inside = sum(
                    box_counts[box_idx]
                    for box_idx in box_idxs_inside
                )
                num_cells_outside = sum(
                    size - box_counts[box_idx]
                    for box_idx in box_idxs_outside
                )

                if num_cells_inside == num_cells_outside:

                    idxs_cells_inside = [
                        (i1, i2)
                        for box_idx in box_idxs_inside
                        for (i1, i2) in instance.idxs_for_dims["box"][box_idx]
                        if instance.get_idx_for_dim(dim, i1, i2) in region
                    ]
                    assert len(idxs_cells_inside) == num_cells_inside

                    idxs_cells_outside = [
                        (i1, i2)
                        for box_idx in box_idxs_outside
                        for (i1, i2) in instance.idxs_for_dims["box"][box_idx]
                        if instance.get_idx_for_dim(dim, i1, i2) not in region
                    ]
                    assert len(idxs_cells_outside) == num_cells_outside

                    application_layout = (
                        dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside
                    )
                    application_layouts.append(application_layout)

    return application_layouts


def _get_idxs_for_region(instance, dim, region):
    idxs = list(itertools.chain.from_iterable(
        instance.idxs_for_dims[dim][idx_dim]
        for idx_dim in region
    ))
    return idxs


# TODO This can be stored in the instance to make it more efficient, but at least this is already an improvement over
#  reprocessing the idxs at every search for applications
def _find_key_boxes_layout(instance):
    key = '\n'.join([
        ' '.join(map(str, [
            instance.get_idx_box(i1, i2)
            for i2 in range(instance.size)
        ]))
        for i1 in range(instance.size)
    ])
    return key


# TODO At some point we should refactor the techniques and add the instance by default to all of them
# TODO When the instance is given, we can remove the redundant argument "chars"
# TODO Move instance back to the first argument, after this has been restructured
def _find_leftovers(options, chars, instance, number_inside, show_logs=False):

    # Differently from the techniques only available for instances with default boxes layout, this technique is always
    #  applied but returns an empty result when the instance uses a default boxes layout;

    assert number_inside <= MAX_NUMBER_INSIDE

    # Initialise details
    details_with_size = []

    # Preprocess idxs or reuse preprocessed idxs
    key_boxes_layout = _find_key_boxes_layout(instance)
    application_layouts = cache_application_layouts.get(key_boxes_layout)
    if not application_layouts:
        application_layouts = _preprocess_layout(instance)
        cache_application_layouts[key_boxes_layout] = application_layouts

    for application_layout in application_layouts:
        dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside = application_layout

        # Special case: Also check for applications with a more than these number of cells inside/outside the region
        # TODO Make this more dynamic
        if number_inside == MAX_NUMBER_INSIDE:
            if num_cells_inside < number_inside:
                continue
        else:
            if num_cells_inside != number_inside:
                continue

        # Determine options inside and outside, which include filled values
        options_inside = set(
            itertools.chain.from_iterable(
                options[i1][i2]
                for (i1, i2) in idxs_cells_inside
            )
        )
        options_inside.update(
            instance[i1][i2]
            for (i1, i2) in idxs_cells_inside
            if instance[i1][i2] != EMPTY_CHAR
        )
        options_outside = set(
            itertools.chain.from_iterable(
                options[i1][i2]
                for (i1, i2) in idxs_cells_outside
            )
        )
        options_outside.update(
            instance[i1][i2]
            for (i1, i2) in idxs_cells_outside
            if instance[i1][i2] != EMPTY_CHAR
        )

        # Only consider applications valid when they actually remove options, as the same applications are
        #  found every iteration as they only depend on the boxes layout which is not modified throughout
        #  the run - which does not clutter the details, does not clog the logs, and speeds up the process
        chars_outside_but_not_inside = options_outside.difference(options_inside)
        chars_inside_but_not_outside = options_inside.difference(options_outside)

        if chars_outside_but_not_inside or chars_inside_but_not_outside:
            # Note: Including the contained options in the application name is needed to make sure the application name
            #  is unique, even if an application using the same region is applied multiple times in the process, which
            #  can even happen in one cleanup process, as newly removed options might enable another application using
            #  the same region but different options;
            name_application = f"\"leftovers-{number_inside}\" in {dim}(s) {tuple(map(lambda x: x + 1, region))} with " \
                f"the cells of box(es) {tuple(map(lambda x: x + 1, box_idxs_inside))} inside and " \
                f"the cells of box(es) {tuple(map(lambda x: x + 1, box_idxs_outside))} outside the region, " \
                f"containing options {tuple(sorted(options_inside))} and {tuple(sorted(options_outside))}"
            if show_logs:
                print(f"Found a valid application of {name_application}")

            removed_chars = []

            # Remove the options occurring outside but not inside
            if len(chars_outside_but_not_inside) > 0:
                if show_logs:
                    print(f"Remove options {chars_outside_but_not_inside} from cells outside the region as they do not occur inside the region")
            for char in chars_outside_but_not_inside:
                check_is_option_removed = False
                for (i1, i2) in idxs_cells_outside:
                    if char in options[i1][i2]:
                        if show_logs:
                            print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                        removed_chars.append(((i1, i2), char))
                        check_is_option_removed = True
                assert check_is_option_removed

            # Remove the options occurring inside but not outside
            if len(chars_inside_but_not_outside) > 0:
                if show_logs:
                    print(f"Remove options {chars_inside_but_not_outside} from cells inside the region as they do not occur outside the region")
            for char in chars_inside_but_not_outside:
                check_is_option_removed = False
                for (i1, i2) in idxs_cells_inside:
                    if char in options[i1][i2]:
                        if show_logs:
                            print(f"Remove '{char}' from {(i1 + 1, i2 + 1)}")
                        removed_chars.append(((i1, i2), char))
                        check_is_option_removed = True
                assert check_is_option_removed

            # Update details
            # TODO The options should not be given here in the details, but passed always to the collection of
            #  details, as they are not modified throughout the process
            application_details = (
                dim, region, box_idxs_inside, box_idxs_outside, idxs_cells_inside, idxs_cells_outside, num_cells_inside,
                options_inside, options_outside, options,
            )

            # TODO Should there be a more detailed priority for when one box has many cells inside/outside?
            details_size = num_cells_inside

            details_with_size.append(((name_application, application_details, removed_chars), details_size))

    # TODO How to order applications? By total inside/outside, or the box with the max number of cells inside/
    #  outside?

    # Select the application with the smallest number of in/out which removes at least one option
    # Note: There are two other techniques for which the final details are selected after sorting: remote-pairs and
    #  ab-chains - in both cases a valid application always removes an option AND finds a new value, by how it is
    #  currently implemented; Here a valid application might not remove any options, so we have to discard these;

    sorted_details = [
        (_details, _size)
        for _details, _size in sorted(details_with_size, key=lambda item: item[-1])
    ]
    # TODO After separating applications for different sizes this should not be necessary anymore
    assert all(_size == number_inside for _details, _size in sorted_details)
    assert sorted_details == details_with_size

    # TODO When we have improved the logic of removing options for applications in algo_human, we should only sort the
    #  found applications here, after which the engine will handle what to do with them; We can even keep applications
    #  which do not remove options, and discard them when processing them (only remove the options for the first
    #  application in the sorted list which removes at least one option)
    sorted_details_with_removed_options = [
        _details
        for _details, _size in sorted_details
        if len(_details[-1]) > 0
    ]
    # Now only applications which remove options are considered valid
    assert len(sorted_details_with_removed_options) == len(sorted_details)

    # Temporarily check that no valid applications are found for instances with a default boxes layout
    # Update: There are valid applications, but apparently there can be no options removed
    if not instance.uses_custom_boxes_layout:
        # assert len(details_with_size) == 0
        assert len(sorted_details_with_removed_options) == 0

    details = sorted_details_with_removed_options[:1]

    # Log the application selection process
    if show_logs:
        print("Number of valid applications:", len(details_with_size))
        # print("Number of valid applications removing options:", len(sorted_details_with_removed_options))
        print("Final selected application:", details)

    return details


def find_leftovers_1(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=1, show_logs=show_logs)


def find_leftovers_2(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=2, show_logs=show_logs)


def find_leftovers_3(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=3, show_logs=show_logs)


def find_leftovers_4(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=4, show_logs=show_logs)


def find_leftovers_5(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=5, show_logs=show_logs)


def find_leftovers_6(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=6, show_logs=show_logs)


def find_leftovers_7(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=7, show_logs=show_logs)


def find_leftovers_8(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=8, show_logs=show_logs)


def find_leftovers_9(options, chars, instance, show_logs=False):
    return _find_leftovers(options, chars, instance, number_inside=9, show_logs=show_logs)
