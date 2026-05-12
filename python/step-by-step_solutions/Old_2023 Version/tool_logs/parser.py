
from tool_logs.input import read_instance, read_list_of_instances


class SyntaxInvalidException(Exception):
    pass


class InputsInvalidException(Exception):
    pass


def process_args(args):

    try:
        assert len(args) > 0, "No arguments provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    try:
        assert len(args) <= 2, "Incorrect number of arguments provided"
    except AssertionError as e:
        raise SyntaxInvalidException(str(e))

    if len(args) == 1:

        try:

            # Arg 1: File name of instance
            file_name = args[0]
            assert file_name.endswith(".xlsx"), "Input file should be a .xlsx file"

        except AssertionError as e:
            raise InputsInvalidException(str(e))

        try:

            instance = read_instance(file_name)

        except Exception as e:
            raise InputsInvalidException(str(e))

        instance_id = '.'.join(file_name.split('/')[-1].split('.')[:-1])

        instances = [(instance_id, instance)]

    else:

        # Arg 1: Flag
        flag_input_list = args[0]
        assert flag_input_list == "--list", f"Flag {flag_input_list} not recognised"

        try:

            # Arg 2: File name of instance
            file_name = args[1]
            assert file_name.endswith(".xlsx"), "Input file should be a .xlsx file"

        except AssertionError as e:
            raise InputsInvalidException(str(e))

        try:

            instances = read_list_of_instances(file_name)

        except Exception as e:
            raise InputsInvalidException(str(e))

    data_folder = file_name[:file_name.rindex('/') + 1] if '/' in file_name else ""

    return data_folder, instances
