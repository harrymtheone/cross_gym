"""Dictionary utilities for configuration management."""

from typing import Any


def class_to_dict(obj: Any) -> dict:
    """Convert a config class to dictionary recursively.
    
    Args:
        obj: Config object to convert
        
    Returns:
        Dictionary representation
    """
    if not hasattr(obj, "__dict__"):
        return obj
    
    result = {}
    for key, value in obj.__dict__.items():
        if key.startswith("_"):
            continue
        if hasattr(value, "__dict__") and not isinstance(value, type):
            result[key] = class_to_dict(value)
        else:
            result[key] = value
    return result


def update_class_from_dict(obj: Any, config_dict: dict) -> None:
    """Update a config object from dictionary.
    
    Args:
        obj: Config object to update
        config_dict: Dictionary with new values
    """
    for key, value in config_dict.items():
        if hasattr(obj, key):
            if isinstance(value, dict) and hasattr(getattr(obj, key), "__dict__"):
                update_class_from_dict(getattr(obj, key), value)
            else:
                setattr(obj, key, value)

