import argparse
import sys

import numpy as np

from my_torch.nn_io import load_network
from my_torch_analyzer.dataset import load_prediction_dataset
from my_torch_analyzer.labels import get_label_from_index

HELP_TEXT = """USAGE
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
DESCRIPTION
--train
Launch the neural network in training mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation and the expected output
separated by space. If specified, the newly trained neural network will be saved in
SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
--predict
Launch the neural network in prediction mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation, and optionally an expected
output.
--save
Save neural network into SAVEFILE. Only works in train mode.
LOADFILE
File containing an artificial neural network
CHESSFILE
File containing chessboards"""


def main() -> int:
    """
    Main entry point for the my_torch_analyzer CLI.
    """
    if "-h" in sys.argv or "--help" in sys.argv:
        print(HELP_TEXT)
        return 0

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save", type=str)
    parser.add_argument("loadfile", nargs="?")
    parser.add_argument("chessfile", nargs="?")

    try:
        args = parser.parse_args()
    except SystemExit:
        return 1

    if not args.loadfile or not args.chessfile:
        print(HELP_TEXT.split("\n")[0])
        print(HELP_TEXT.split("\n")[1])
        return 1

    if args.predict and args.train:
        print("Error: --predict and --train are mutually exclusive.")
        return 1

    if not args.predict and not args.train:
        print("Error: You must specify either --predict or --train.")
        return 1

    if args.predict and args.save:
        print("Error: --save is only valid in --train mode.")
        return 1

    if args.predict:
        try:
            network = load_network(args.loadfile)
            inputs = load_prediction_dataset(args.chessfile)
            inputs = inputs.reshape(inputs.shape[0], -1)

            outputs = network.forward(inputs)

            predicted_indices = np.argmax(outputs, axis=1)

            for idx in predicted_indices:
                print(get_label_from_index(idx))

        except Exception as e:
            print(f"Error during prediction: {e}")
            return 1

    elif args.train:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
