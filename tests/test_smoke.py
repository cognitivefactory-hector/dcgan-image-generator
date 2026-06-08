"""M0 placeholder so `pytest` collects and CI goes green.

The real mechanics tests arrive in M1 (`test_shapes.py`: G/D I/O shapes;
`test_train_step.py`: one alternating step produces finite losses). Those import
torch, so M1 also updates CI to install the package. Keep this test torch-free.
"""

import sys


def test_python_is_312_plus():
    assert sys.version_info >= (3, 12)
