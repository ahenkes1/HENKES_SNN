"""Code of the publication 'xxx' published in
https://doi.org/10.1016/xxx
by Alexander Henkes and Henning Wessels from TU Braunschweig and
Jason K. Eshraghian from University of California, Santa Cruz.
"""
import argparse
import numpy as np
import random
import torch

import experiment_plasticity


# Seeds
SEED = 42
# Python RNG
random.seed(SEED)
# Numpy RNG
np.random.seed(SEED)
# TF RNG
torch.manual_seed(SEED)


def get_input():
    long_description = str(
        "Code of the publication 'xxx' published in"
        "https://doi.org/10.1016/xxx"
        "by Alexander Henkes and Henning Wessels "
        "from TU Braunschweig and"
        "Jason K. Eshraghian from University of California, Santa Cruz.S."
    )

    parser = argparse.ArgumentParser(
        description=long_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--example",
        default="plasticity",
        type=str,
        choices=["elasticity", "plasticity"],
        help="Execute the numerical examples from the paper",
    )
    arguments = parser.parse_args()
    return arguments


def main(parser_args=None):
    """Main function for all numerical examples."""
    EXAMPLE = parser_args.example

    if EXAMPLE == "elasticity":
        raise NotImplementedError("This experiment does not exist.")

    elif EXAMPLE == "plasticity":
        experiment_plasticity.main()

    else:
        raise NotImplementedError("This experiment does not exist.")

    return None


if __name__ == "__main__":
    args = get_input()
    main(parser_args=args)
