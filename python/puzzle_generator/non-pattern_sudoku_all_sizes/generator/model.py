
import itertools

from generator.boxes import create_default_layout_boxes


DIMENSIONS = ["row", "col", "box"]

EMPTY_CHAR = ' '

DUMMY_CHAR = '*'

SIZES_REQUIRING_STANDARD_LAYOUT = [6, 8]


def count_empty_cells(instance):
    return sum(e == EMPTY_CHAR for row in instance for e in row)


def count_non_empty_cells(instance):
    return sum(e != EMPTY_CHAR for row in instance for e in row)


# TODO Improve copy functionality:
#  - Don't preprocess idxs every time
#  - Either pass all arguments or none
def copy_instance(instance):
    l_copy = list(map(lambda l: l.copy(), instance))
    # is_chars=False is used for efficiency, after initialising the object the chars are copied
    copy = Instance(l_copy, is_chars=False, preprocessed_dims=instance.dims)
    copy.chars = instance.chars
    # TODO Separate idxs from values and copy idxs to copied object instead of re-preprocessing them
    # TODO Not only is it more efficient to preprocess idxs only once, it is also cleaner and avoids having to override
    #  values such as the one below
    # copy.uses_standard_boxes_layout = instance.uses_standard_boxes_layout
    return copy


def is_only_char(instance, i1, i2):
    char = instance[i1][i2]
    assert char != EMPTY_CHAR
    count = sum(e == char for row in instance for e in row)
    return count == 1


class Instance(list):

    # TODO Remove layout from arguments, keep only one format
    def __init__(self, l, is_chars=True, layout=None, layout_boxes=None, preprocessed_dims=None):
        super().__init__(l)
        size = len(l)
        assert len(self) == size and all(len(row) == size for row in self), "Not a valid instance!"
        self.size = size
        if is_chars:
            chars = self._get_chars(size)
            assert len(chars) == size and EMPTY_CHAR not in chars, f"Passed chars not valid: {chars}"
            self.chars = chars

        # By default, the idxs are preprocessed upon instantiation, but this can be suppressed when copying an instance
        if preprocessed_dims is None:
            dims = preprocess_dims(size, layout, layout_boxes)
        else:
            # Can't specify both optional layout arguments and already preprocessed dims
            assert layout is None and layout_boxes is None
            dims = preprocessed_dims
        self.dims = dims
        (size, layout, (box_width, box_height), layout_boxes, uses_custom_boxes_layout, (idx_cell_to_idx_box, idxs_for_dims)) = dims
        self.layout = layout
        self.box_width = box_width
        self.box_height = box_height
        self.layout_boxes = layout_boxes
        self.uses_custom_boxes_layout = uses_custom_boxes_layout
        self.idx_cell_to_idx_box = idx_cell_to_idx_box
        self.idxs_for_dims = idxs_for_dims

    # TODO Use color palette
    def __str__(self):
        result = ""
        for i1, row in enumerate(self):
            result += str(row) + '   ' + str([self.get_idx_box(i1, i2) + 1 for i2 in range(self.size)]) + '\n'
        return result

    def __repr__(self):
        return self.__str__()

    def _get_chars(self, size):
        chars = set(itertools.chain(*self))
        try:
            chars.remove(EMPTY_CHAR)
        except KeyError:
            pass
        # Allow for one dummy character
        if len(chars) == size - 1:
            print(f"WARNING: Missing one character in instance, using dummy char '{DUMMY_CHAR}' instead")
            chars.update({DUMMY_CHAR})
        return chars

    # def get_dimensions(self, i1, i2):
    #     row = self[i1]
    #     col = [row[i2] for row in self]
    #     box = get_box(self, i1, i2)
    #     return row, col, box

    def is_empty(self, i1, i2):
        return self[i1][i2] == EMPTY_CHAR

    # TODO Remove or refer to idxs_for_dims
    def get_rows(self):
        rows = [(i1, self[i1]) for i1 in range(self.size)]
        return rows

    # TODO Remove or refer to idxs_for_dims
    def get_cols(self):
        cols = [(i2, [row[i2] for row in self]) for i2 in range(self.size)]
        return cols

    def copy(self):
        return copy_instance(self)

    def get_values(self, dim, idx):
        return [self[i1][i2] for (i1, i2) in self.idxs_for_dims[dim][idx]]

    def get_idx_box(self, i1, i2):
        return self.idx_cell_to_idx_box[(i1, i2)]

    def get_idx_for_dim(self, dim, i1, i2):
        assert dim in DIMENSIONS
        if dim == "row":
            return i1
        elif dim == "col":
            return i2
        else:
            return self.get_idx_box(i1, i2)

    def get_idxs_in_dim(self, dim, i1, i2):
        return self.idxs_for_dims[dim][self.get_idx_for_dim(dim, i1, i2)]


# TODO We don't want to do this everytime an instance is copied..
def preprocess_dims(size, layout, layout_boxes):

    # Only either a default or custom boxes layout can be specified
    assert (layout is not None) + (layout_boxes is not None) <= 1, "Should only specify either a default or custom layout for boxes"

    # TODO It could be that a standard boxes layout is specified in the file specifying a custom layout, but this should not be used..
    uses_custom_boxes_layout = layout_boxes is not None

    if layout_boxes is not None:
        assert layout is None  # When a custom boxes layout is specified, no default layout should be specified
        box_width, box_height = None, None
    else:
        # If boxes are not of equal width/height, the layout should be specified
        if layout is not None:
            assert layout_boxes is None  # When a default layout is specified, no custom layout should be specified
            box_width, box_height = map(int, layout.split("x"))
        else:
            # This is done to avoid having to rewrite existing code -- It would be better if layout is always given
            try:
                length = [e for e in [2, 3, 4] if e ** 2 == size][0]
            except IndexError:
                raise Exception(f"Could not establish default boxes layout for size {size}")
            box_width = box_height = length
        layout_boxes = create_default_layout_boxes(box_width, box_height)

    # Preprocessing for efficiency purposes
    idx_cell_to_idx_box = {
        (i1, i2): [idx_box for idx_box, idxs_for_box in layout_boxes.items() if (i1, i2) in idxs_for_box][0]
        for (i1, i2) in itertools.product(range(size), repeat=2)
    }
    assert sorted(idx_cell_to_idx_box.values()) == list(itertools.chain.from_iterable(zip(*itertools.repeat(range(size), size))))
    idxs_for_dims = _preprocess_idxs_for_dims(size, layout_boxes)

    # TODO Remove box_width, box_height when the logic for boxes-x techniques is rewritten, which is the only place requiring the box dims

    dims = (size, layout, (box_width, box_height), layout_boxes, uses_custom_boxes_layout, (idx_cell_to_idx_box, idxs_for_dims))

    return dims


def _preprocess_idxs_for_dims(size, layout_boxes):
    # This can be done once as a preprocessing step, after which the values should be accessed dynamically
    idxs_for_dims = {}
    for dim in ["row", "col", "box"]:
        if dim == "row":
            idxs = {i1: [(i1, i2) for i2 in range(size)] for i1 in range(size)}
        elif dim == "col":
            idxs = {i2: [(i1, i2) for i1 in range(size)] for i2 in range(size)}
        elif dim == "box":
            idxs = layout_boxes
        else:
            raise NotImplementedError()
        assert sorted(itertools.chain.from_iterable(idxs.values())) == list(itertools.product(range(size), repeat=2))
        idxs_for_dims[dim] = idxs
    return idxs_for_dims
