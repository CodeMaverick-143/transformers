# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import doctest
import sys
import warnings
from os.path import abspath, dirname, join

import _pytest
import pytest

from transformers.testing_utils import HfDoctestModule, HfDocTestParser


NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_processor",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_save_load_fast_init_from_base",
    "test_fast_init_context_manager",
    "test_fast_init_tied_embeddings",
    "test_save_load_fast_init_to_base",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_pt_tf_model_equivalence",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
    "/agents/",
}

# allow having multiple repository checkouts and not needing to remember to rerun
# `pip install -e '.[dev]'` when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )
    config.addinivalue_line(
        "markers", "is_pt_flax_cross_test: mark test to run only when PT and FLAX interactions are tested"
    )
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "agent_tests: mark the agent tests that are run on their specific schedule")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")

    '''
    Hook programmatically configuring the top-level ``"pytest.ini"`` file.
    '''

    # Programmatically add our custom "run_in_subprocess" mark, enabling tests
    # to notify the pytest_pyfunc_call() hook that they require isolation to a
    # Python subprocess of the current Python process.
    config.addinivalue_line(
        'markers',
        f'run_in_subprocess: mark test to run in an isolated subprocess',
    )

def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)


def pytest_sessionfinish(session, exitstatus):
    # If no tests are collected, pytest exists with code 5, which makes the CI fail.
    if exitstatus == 5:
        session.exitstatus = 0


# Doctest custom flag to ignore output.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker
_pytest.doctest.DoctestModule = HfDoctestModule
doctest.DocTestParser = HfDocTestParser


#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2014-2023 Beartype authors.
# See "LICENSE" for further details.

'''
**Root test configuration** (i.e., early-time configuration guaranteed to be
run by :mod:`pytest` *before* passed command-line arguments are parsed) for
this test suite.

Caveats
----------
For safety, this configuration should contain *only* early-time hooks
absolutely required by :mod:`pytest` design to be defined in this
configuration. Hooks for which this is the case (e.g.,
:func:`pytest_addoption`) are explicitly annotated as such in official
:mod:`pytest` documentation with a note resembling:

    Note

    This function should be implemented only in plugins or ``conftest.py``
    files situated at the tests root directory due to how pytest discovers
    plugins during startup.

This file is the aforementioned ``conftest.py`` file "...situated at the tests
root directory."
'''

# ....................{ IMPORTS                            }....................
from pytest import Function
from typing import Optional

# ....................{ HOOKS ~ ini                        }....................
# def pytest_configure(config) -> None:
#     '''
#     Hook programmatically configuring the top-level ``"pytest.ini"`` file.
#     '''
#
#     # Programmatically add our custom "run_in_subprocess" mark, enabling tests
#     # to notify the pytest_pyfunc_call() hook that they require isolation to a
#     # Python subprocess of the current Python process.
#     config.addinivalue_line(
#         'markers',
#         f'{_MARK_NAME_SUBPROCESS}: mark test to run in an isolated subprocess',
#     )

# ....................{ HOOKS ~ test : run                 }....................
def pytest_pyfunc_call(pyfuncitem: Function) -> Optional[bool]:
    '''
    Hook intercepting the call to run the passed :mod:`pytest` test function.

    Specifically, this test:

    * If this test has been decorated by our custom
      ``@pytest.mark.run_in_subprocess`` marker, runs this test in a Python
      subprocess of the current Python process isolated to this test.
    * Else, runs this test in the current Python process by deferring to the
      standard :mod:`pytest` logic for running this test.

    Parameters
    ----------
    pyfuncitem: Function
        :mod:`pytest`-specific object encapsulating the current test function
        being run.

    Returns
    ----------
    Optional[bool]
        Either:

        * If this hook ran this test, :data:`True`.
        * If this hook did *not* run this test, :data:`None`.

    See Also
    ----------
    https://github.com/ansible/pytest-mp/issues/15#issuecomment-1342682418
        GitHub comment by @pelson (Phil Elson) strongly inspiring this hook.
    '''

    breakpoint()

    # If this test has been decorated by our custom
    # @pytest.mark.run_in_subprocess marker...
    if _MARK_NAME_SUBPROCESS in pyfuncitem.keywords:
        # Defer hook-specific imports.
        from multiprocessing import Process
        from pytest import fail

        def _run_test_in_subprocess() -> object:
            '''
            Run the current :mod:`pytest` test function isolated to a Python
            subprocess of the current Python process.

            Returns
            ----------
            object
                Arbitrary object returned by this test if any *or* :data:`None`.
            '''

            # Defer subpracess-specific imports.
            import sys

            # Monkey-patch the unbuffered standard error and output streams of
            # this subprocess with buffered equivalents, ensuring that pytest
            # will reliably capture *all* standard error and output emitted by
            # running this test.
            sys.stderr = _UnbufferedOutputStream(sys.stderr)
            sys.stdout = _UnbufferedOutputStream(sys.stdout)

            # Run this test and return the result of doing so.
            return pyfuncitem.obj()

        # Python subprocess tasked with running this test.
        test_subprocess = Process(target=_run_test_in_subprocess)

        # Begin running this test in this subprocess.
        test_subprocess.start()

        # Block this parent Python process until this test completes.
        test_subprocess.join()

        # If this subprocess reports non-zero exit status, this test failed. In
        # this case...
        if test_subprocess.exitcode != 0:
            # Human-readable exception message to be raised.
            exception_message = (
                f'Test "{pyfuncitem.name}" failed in isolated subprocess with:')

            # Raise a pytest-compliant exception.
            raise fail(exception_message, pytrace=False)
        # Else, this subprocess reports zero exit status. In this case, this
        # test succeeded.

        # Notify pytest that this hook successfully ran this test.
        return True

    # Notify pytest that this hook avoided attempting to run this test, in which
    # case pytest will continue to look for a suitable runner for this test.
    return None

# ....................{ PRIVATE ~ globals                  }....................
_MARK_NAME_SUBPROCESS = 'run_in_subprocess'
'''
**Subprocess mark** (i.e., name of our custom :mod:`pytest` mark, enabling tests
to notify the :func:`.pytest_pyfunc_call` hook that they require isolation to a
Python subprocess of the current Python process).
'''

# ....................{ PRIVATE ~ classes                  }....................
class _UnbufferedOutputStream(object):
    '''
    **Unbuffered standard output stream** (i.e., proxy object encapsulating a
    buffered standard output stream by forcefully flushing that stream on all
    writes to that stream).

    See Also
    ----------
    https://github.com/ansible/pytest-mp/issues/15#issuecomment-1342682418
        GitHub comment by @pelson (Phil Elson) strongly inspiring this class.
    '''

    def __init__(self, stream) -> None:
        self.stream = stream

    def write(self, data) -> None:
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas) -> None:
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr: str) -> object:
        return getattr(self.stream, attr)