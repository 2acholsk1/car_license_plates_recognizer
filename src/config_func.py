#!/usr/bin/env python3
"""Config fuction implementation

Returns:
    _type_: config
"""
import yaml


def load_config(filename):
    """Function to load .yaml files

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

