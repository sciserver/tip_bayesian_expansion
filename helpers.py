# MIT License

# Copyright (c) 2023 sciserver

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Helper functions to run the bayesian expansion algorithm."""


import itertools
import logging
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse


def get_sql(path: str, **format_keys: Dict[str, Any]) -> str:
    """Helper function to read in sql files and format them with the given keys.

    Args:
        path (str): The path to the sql file
        format_keys (Dict[str, Any]): The keys to format the sql text with

    Returns:
        str: The formatted sql text
    """
    with open(path, "r") as f:
        return "".join(l for l in f.readlines()).format(**format_keys)


def transform_row_to_idx_pairs(
    id_idx: Dict[int, int], coauthors: str
) -> List[Tuple[int, int]]:
    """Transform a row of the coauthor matrix into a list of coauthor idx pairs.

    Args:
        id_idx (Dict[int, int]): A dictionary mapping author ids to their index
            in the coauthor matrix
        coauthors (str): A string of comma separated author ids

    Returns:
        List[Tuple[int, int]]: A list of coauthor idx pairs
    """

    coauthors_list = coauthors.split(", ")
    coauthor_ids = list(map(int, coauthors_list))
    coauthor_idxs = list(map(lambda x: id_idx[x], coauthor_ids))
    coauthor_idx_pairs = list(itertools.combinations(coauthor_idxs, 2))

    return coauthor_idx_pairs


def arr_to_matrix(
    transform_f: Callable[[str], List[Tuple[int, int]]],
    n: int,
    split_idx: int,
    arr: np.ndarray,
) -> sparse.spmatrix:
    """Transform an array of coauthor strings into a coauthor matrix.

    Args:
        transform_f (Callable[[str], List[Tuple[int, int]]]): A function that
            transforms a coauthor string into a list of coauthor idx pairs
        n (int): The number of authors
        split_idx (int): The index to split the matrix at
        arr (np.ndarray): An array of coauthor strings

    Returns:
        sparse.spmatrix: The coauthor matrix
    """

    values, row_idxs, col_idxs = [], [], []
    for row_idx, col_idx in itertools.chain.from_iterable(
        map(
            transform_f,
            filter(lambda s: "," in s, arr),  # get all papers with more than one author
        )
    ):
        values.append(True)
        row_idxs.append(row_idx)
        col_idxs.append(col_idx)

    A = sparse.coo_array(
        # (data, (row, col))
        (values, (row_idxs, col_idxs)),
        dtype=bool,
        shape=(n, n),
    ).tocsr()

    return (A + A.T)[split_idx:] if split_idx else (A + A.T)


def update(
    f: np.ndarray,  # [N,]
    A: np.ndarray,  # [U, N]
    labeled_idx: int,  # [0,]
    prior_α: float,  # [0,]
    prior_β: float,  # [0,]
) -> np.ndarray:
    """Update the f vector.

    Args:
        f (np.ndarray): The f vector
        A (np.ndarray): The coauthor matrix
        labeled_idx (int): The index to split the f vector at
        prior_α (float): The prior α for the beta distribution
        prior_β (float): The prior β for the beta distribution

    Returns:
        np.ndarray: The updated f vector
    """

    N = len(f)
    numerator = prior_α + A.dot(f)  # [U,]
    denominator = prior_α + prior_β + A.sum(axis=1)  # [U,]
    new_f = numerator / denominator
    return np.concatenate([f[:labeled_idx], new_f])


def compute(
    f: np.ndarray,
    A: np.ndarray,
    labeled_idx: int,
    prior_α: float,
    prior_β: float,
    iter_tolerance: float = 1e-5,
    max_iter: int = 1000,
    logger=None,
) -> np.ndarray:
    """Runs the update algorithms iteratively until convergence or `max_iter`.

    Args:
        f (np.ndarray): The f vector
        A (np.ndarray): The coauthor matrix
        labeled_idx (int): The index to split the f vector at
        prior_α (float): The prior α for the beta distribution
        prior_β (float): The prior β for the beta distribution
        iter_tolerance (float, optional): The tolerance for convergence. Defaults to 1e-5.
        max_iter (int, optional): The maximum number of iterations to run. Defaults to 1000.
        logger ([type], optional): The logger instance to use. Defaults to None.


    Returns:
        np.ndarray: The updated f vector
    """

    f[labeled_idx:] = np.random.uniform(size=len(f) - labeled_idx)
    update_diffs = []
    for i in range(max_iter):
        new_f = update(f, A, labeled_idx, prior_α, prior_β)
        err = np.linalg.norm(new_f - f)
        update_diffs.append(err)
        f = new_f

        if err < iter_tolerance:
            if logger:
                logger.info(f"converged in {i} steps")
            break

    return f


class LogTime:
    """A context manager class for logging code timings and errors.

    This class will log the time it takes to run some code along with any
    exceptions that occur within in the context. Timings are logged at the INFO
    level and errors are logged at the FATAL level.

    Example:

    >>> import logging
    >>> logger = logging.getLogger("example")
    >>>
    >>> with LogTime(logger, "Calculating Sum"):
    >>>     a = 2 + 2

    The log file will look something like (depending on your formatting):

    >>> Starting Calculating Sum
    >>> Completed in 7.62939453125e-06 seconds
    """

    def __init__(self, logger: logging.Logger, task_str: str):
        """A context manager class for logging code timings and errors.

        Args:
            logger (logging.Logger): The logger instance to use for logging
            task_str (str): The string describing the section of code being run

        """
        self.logger = logger
        self.task_str = task_str

    def __enter__(self):
        self.logger.info(f"Starting {self.task_str}")
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.logger.info(f"Completed in {time.time() - self.start} seconds")
        else:
            self.logger.fatal(f"{exc_type}\n{exc_value}\n{traceback}")
