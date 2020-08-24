from .structured_data.classification.sklearn import create_project as sklearn_classification
from .structured_data.regression.pytorch import create_project as pytorch_regression
from .structured_data.classification.pytorch import create_project as pytorch_classification

import argparse


def main():
    func_map = {
        "pytorch-structured-classification": pytorch_classification.create_project,
        "sklearn-structured-classification": sklearn_classification.create_project,

        "pytorch-structured-regression": pytorch_regression.create_project,
    }

    parser = argparse.ArgumentParser(description="Tool to help the creation of TanD projects")

    parser.add_argument("--template", type=str, required=True, choices=[
        "pytorch-structured-classification",
        "sklearn-structured-classification",
        "pytorch-structured-regression"
    ])

    args = parser.parse_args()

    print(args)
    print(f"Creating TanD project of type {args.template}")
    func_map[args.template]()


if __name__ == '__main__':
    print(__name__)
    main()
