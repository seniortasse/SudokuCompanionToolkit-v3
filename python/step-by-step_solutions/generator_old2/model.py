
import itertools


EMPTY_CHAR = ' '

DUMMY_CHAR = '*'


def count_empty_cells(instance):
    return sum(e == EMPTY_CHAR for row in instance for e in row)


def count_non_empty_cells(instance):
    return sum(e != EMPTY_CHAR for row in instance for e in row)


def get_box(instance, i1, i2):
    box_height, box_width = instance.box_height, instance.box_width
    b1 = i1 // box_height
    b2 = i2 // box_width
    box = [
        instance[_i1][_i2]
        for _i1 in range(b1 * box_height, (b1 + 1) * box_height)
        for _i2 in range(b2 * box_width, (b2 + 1) * box_width)
    ]
    return box


def copy_instance(instance):
    l_copy = list(map(lambda l: l.copy(), instance))
    copy = Instance(l_copy, layout=instance.layout, is_chars=False)
    copy.chars = instance.chars
    copy.box_width = instance.box_width
    copy.box_height = instance.box_height
    copy.size = instance.size
    return copy


def fits(instance, char, i1, i2, check_row=True, check_col=True, check_box=True):
    if instance[i1][i2] != EMPTY_CHAR:
        return False
    if check_row:
        row = instance[i1]
        if char in row:
            return False
    if check_col:
        col = [row[i2] for row in instance]
        if char in col:
            return False
    if check_box:
        box = get_box(instance, i1, i2)
        if char in box:
            return False
    return True


def is_only_char(instance, i1, i2):
    char = instance[i1][i2]
    assert char != EMPTY_CHAR
    count = sum(e == char for row in instance for e in row)
    return count == 1


class Instance(list):

    def __init__(self, l, layout=None, is_chars=True):
        super().__init__(l)
        size = len(l)
        if is_chars:
            assert len(self) == size and len(self[0]) == size, "Not a valid instance!"
            chars = self._get_chars(size)
            assert len(chars) == size and not EMPTY_CHAR in chars, f"Passed chars not valid: {chars}"
            self.chars = chars
        self.layout=layout
        # If boxes are not of equal width/height, the layout should be specified
        if layout is not None:
            box_width, box_height = map(int, layout.split("x"))
        else:
            # This is done to aoid having to rewrite existing code -- It would be better if layout is always given
            length = [e for e in [2, 3, 4, 5] if e ** 2 == size][0]
            box_width = box_height = length
        self.box_width = box_width
        self.box_height = box_height
        self.size = size

    def __str__(self):
        result = ""
        for row in self:
            result += str(row) + '\n'
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

    def get_dimensions(self, i1, i2):
        row = self[i1]
        col = [row[i2] for row in self]
        box = get_box(self, i1, i2)
        return row, col, box

    def is_empty(self, i1, i2):
        return self[i1][i2] == EMPTY_CHAR

    def get_rows(self):
        rows = [(i1, self[i1]) for i1 in range(self.size)]
        return rows

    def get_cols(self):
        cols = [(i2, [row[i2] for row in self]) for i2 in range(self.size)]
        return cols

    def get_boxs(self):
        boxs = [
            ((b1, b2), get_box(self, b1 * self.box_height, b2 * self.box_width))
            for b1 in range(self.size // self.box_height)
            for b2 in range(self.size // self.box_width)
        ]
        return boxs

    def copy(self):
        return copy_instance(self)

    def get_box(self, b1, b2):
        box = [
            [
                self[i1][i2]
                for i2 in range(b2 * self.box_width, (b2 + 1) * self.box_width)
            ]
            for i1 in range(b1 * self.box_height, (b1 + 1) * self.box_height)
        ]
        return box

