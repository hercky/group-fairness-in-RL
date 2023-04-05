class EpisodeDoneError(TimeoutError):
    """An error for when the episode is over."""
    pass


class InvalidActionError(ValueError):
    """An error for when an invalid action is taken"""
    pass


class InvalidArgumentException(Exception):
    """Exception raised for errors due to misspecified input arguments.

    Attributes:
        input_attr -- input due to which the exception was raised
        message -- explanation of the error
    """

    def __init__(self, input_attr, message="Invalid input argument given!"):
        self.input_attr = input_attr
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.input_attr} -> {self.message}'

