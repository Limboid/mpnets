from exports import export
import glom


@export
def is_valid_glom_string(obj, glom_str):
    try:
        glom.glom(obj, glom_str)
        return True
    except:
        return False


@export
def error(msg):
    raise Exception(msg)


@export
def unsqueeze_list(x):
    try:
        if len(x) == 1:
            return x[0]
    except:
        pass
    return x


@export
def select(dict, keys):
    """
    Selects a subset of keys from a dictionary.

    Args:
        dict: A dictionary.
        keys: A list of keys to select.

    Returns:
        A dictionary with the selected keys.

    Example:
        >>> select({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
        {'a': 1, 'c': 3}

    Raises:
        KeyError: If a key is not in the dictionary.
    """
    return {k: dict[k] for k in keys}
