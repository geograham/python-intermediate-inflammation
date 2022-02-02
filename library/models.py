import numpy as np

class Book:
    """A collection of bound pages of text."""

    def __init__(self,name,author):
        self.name=name
        self.author=author

    def __str__(self):
        return self.name + 'by' + self.author 