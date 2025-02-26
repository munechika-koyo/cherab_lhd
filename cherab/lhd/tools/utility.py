"""Module containing utility functions to help with data processing and analysis."""

import json

import numpy as np

__all__ = ["sanitize_dict"]


def sanitize_dict(attrs: dict[str, object]) -> dict[str, object]:
    """Convert dictionary to a form that can be saved in NetCDF or JSON.

    This function converts all values in the input dictionary to basic Python types,
    such as str, int, float, bool, or list.
    This is useful when saving dictionaries to NetCDF or JSON files, as these formats do not support
    all Python types, such as NumPy types or custom objects.

    Parameters
    ----------
    attrs : dict
        Dictionary of attributes to sanitize.

    Returns
    -------
    dict[str, object]
        Sanitized dictionary of attributes.

    Examples
    --------
    >>> x = {
    ...     "str": "string",
    ...     "int": 1,
    ...     "float": 1.0,
    ...     "bool": True,
    ...     "np_int": np.int64(1),
    ...     "np_float": np.float64(1.0),
    ...     "np_bool": np.bool_(True),
    ...     "list": [1, 2, 3],
    ...     "dict": {"key": "value"},
    ...     "none": None,
    ...     "object": object(),
    ... }
    >>> sanitize_dict(x)
    {
        "str": "string",
        "int": 1,
        "float": 1.0,
        "bool": "True",
        "np_int": 1,
        "np_float": 1.0,
        "np_bool": "True",
        "list": [1, 2, 3],
        "dict": '{"key": "value"}',
        "none": "None",
        "object": "<object object at 0x...>",
    }
    """

    def convert_value(value):
        if isinstance(value, bool):  # Put first because bool is a subclass of int
            return str(value)  # Convert bool to string
        elif isinstance(value, (str, int, float)):
            return value
        elif isinstance(value, np.bool_):
            return str(value.item())
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()  # Convert NumPy types to Python basic types
        elif isinstance(value, list):
            return value  # Leave lists as they are
        elif isinstance(value, dict):
            return json.dumps(sanitize_dict(value))  # Recursively sanitize nested dictionaries
        elif value is None:
            return "None"  # Convert None to string
        else:
            return str(value)  # Convert other objects to string

    return {key: convert_value(value) for key, value in attrs.items()}
