def is_number(maybe_number_str: str) -> bool:
    """Check whether a given string is a number. We do not consider special cases such as 'infinity' and 'nan',
    which technically are floats. We do consider, however, floats like '1.23'.
    :param maybe_number_str: a string that might be a number
    :return: whether the given number is indeed a number
    """
    if maybe_number_str in ["infinity", "nan", "inf"]:
        return False

    try:
        float(maybe_number_str)
        return True
    except ValueError:
        return False
