"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])

def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[1, 2, 3], [3, 4, 5], [5, 6, 7]], [5, 6, 7]),
        ([[1, 2, -3], [3, -4, -5], [-5, -6, -7]], [3, 2, -3]),
    ])

def test_daily_max(test, expected):
    """Test max function works for array of zeroes, positive integers, and positive and negative integers."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[1, 2, 3], [3, 4, 5], [5, 6, 7]], [1, 2, 3]),
        ([[1, 2, -3], [3, -4, -5], [-5, -6, -7]], [-5, -6, -7]),
    ])

def test_daily_min(test, expected):
    """Test min function works for array of zeroes, positive integers, and positive and negative integers."""
    from inflammation.models import daily_min
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

# TODO(lesson-robust) Implement tests for the other statistical functions
