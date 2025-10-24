"""Configuration class decorator for dataclasses."""

from dataclasses import dataclass


def configclass(cls, **kwargs):
    """Decorator for configuration dataclasses.
    
    This is a convenience wrapper around @dataclass that adds
    common functionality for configuration classes.
    
    Args:
        cls: The class to decorate
        **kwargs: Additional arguments passed to dataclass
        
    Returns:
        Decorated class with dataclass functionality
    """
    # Set default kwargs for config classes
    default_kwargs = {
        'frozen': False,  # Allow mutation
    }
    default_kwargs.update(kwargs)
    
    return dataclass(cls, **default_kwargs)
