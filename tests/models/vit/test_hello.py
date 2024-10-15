import pytest
import unittest
import os

main_pid = os.getpid()

# @pytest.mark.run_in_subprocess
# class TestHello:
#
#     def test_foo(self):
#         assert 1 != 2
#         assert os.getpid() != main_pid


@pytest.mark.run_in_subprocess
def test_foo_2():
    assert 1 != 2
    assert os.getpid() != main_pid
