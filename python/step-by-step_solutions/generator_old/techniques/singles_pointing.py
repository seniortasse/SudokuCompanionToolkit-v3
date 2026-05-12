

def find_pointing_singles(options, chars, show_logs=False):

    # Adaptation of the previous version applied to options

    box_height, box_width, size = options.box_height, options.box_width, options.size

    details = []

    # Identify rays
    for b1 in range(size // box_height):
        for b2 in range(size // box_width):
            # if (b1, b2) == (1, 2):
            #     raise Exception()
            for char in chars:
                _options = []
                for _i1 in range(box_height):
                    for _i2 in range(box_width):
                        i1 = b1 * box_height + _i1
                        i2 = b2 * box_width + _i2
                        # Apply technique based on original (not updated during the process!) options
                        if char in options[i1][i2]:
                            _options.append((i1, i2))

                # Identify row rays
                row_idxs = set(option[0] for option in _options)
                if len(row_idxs) == 1:
                    row_idx = row_idxs.pop()
                    name_application = f"hor ray for '{char}' in box {(b1 + 1, b2 + 1)} at row {row_idx + 1}"
                    if show_logs:
                        print(f"Identified {name_application}")

                    removed_chars = []
                    for i2 in range(size):
                        # Only remove options for cols outside the box
                        if not b2 * box_width <= i2 < (b2 + 1) * box_width:
                            if char in options[row_idx][i2]:
                                # options[row_idx][i2].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(row_idx + 1, i2 + 1)}")
                                removed_chars.append(((row_idx, i2), char))

                    details.append((name_application, (char, (b1, b2), _options, "hor", row_idx), removed_chars))

                # Identify col rays
                col_idxs = set(option[1] for option in _options)
                if len(col_idxs) == 1:
                    col_idx = col_idxs.pop()
                    name_application = f"ver ray for '{char}' in box {(b1 + 1, b2 + 1)} at col {col_idx + 1}"
                    if show_logs:
                        print(f"Identified {name_application}")

                    removed_chars = []
                    for i1 in range(size):
                        # Only remove options for rows outside the box
                        if not b1 * box_height <= i1 < (b1 + 1) * box_height:
                            if char in options[i1][col_idx]:
                                # options[i1][col_idx].remove(char)
                                if show_logs:
                                    print(f"Remove {char} from {(i1 + 1, col_idx + 1)}")
                                removed_chars.append(((i1, col_idx), char))

                    details.append((name_application, (char, (b1, b2), _options, "ver", col_idx), removed_chars))

    return details
