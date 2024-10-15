"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 8]], [4, 6, 8]),
        ([[6, -2, 5], [1, -6, 2], [-4, -1, 9]], [6, -1, 9]),
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeroes and positive integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ])
def test_daily_min(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


@pytest.mark.parametrize('data, expected_standard_deviation', [
    ([0, 0, 0], 0.0),
    ([1.0, 1.0, 1.0], 0),
    ([0.0, 2.0], 1.0)
])
def test_daily_standard_deviation(data, expected_standard_deviation):
    from inflammation.models import s_dev
    result_data = s_dev(data)['standard deviation']
    npt.assert_approx_equal(result_data, expected_standard_deviation)
