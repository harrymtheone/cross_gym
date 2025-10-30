"""Helper utility functions."""

from typing import Any, Dict


def class_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a class instance to a dictionary.
    
    Args:
        obj: Class instance to convert
        
    Returns:
        Dictionary representation of the class
    """
    if not hasattr(obj, '__dict__'):
        return obj

    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith('_'):
            continue
        if hasattr(val, '__dict__'):
            result[key] = class_to_dict(val)
        else:
            result[key] = val

    return result
