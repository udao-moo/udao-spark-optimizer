class NoBenchmarkError(ValueError):
    """raise when no valid benchmark is found"""


class NoQTypeError(ValueError):
    """raise when no valid mode is found (only q and qs)"""


class SolutionNotFoundError(ValueError):
    """raise when no solution is found"""
