import argparse
import sys

import numpy as np

from my_torch.losses import cross_entropy_grad, cross_entropy_loss
from my_torch.nn_io import load_network, save_network
from my_torch.optimizers import SGD
from my_torch.training import train, train_validation_split
from my_torch_analyzer.dataset import load_dataset, load_prediction_dataset
from my_torch_analyzer.labels import get_label_from_index

EXIT_SUCCESS = 0
EXIT_ERROR = 84

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


class SubjectArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that exits with code 84 on error, as required by the subject.
    """

    def error(self, message):
        print(HELP_TEXT)
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


def run_analyzer(args: argparse.Namespace) -> None:
    """
    Core logic for handling the analyzer commands.
    """
    if args.predict and args.train:
        raise ValueError("--predict and --train are mutually exclusive.")

    if not args.predict and not args.train:
        raise ValueError("You must specify either --predict or --train.")

    if args.predict and args.save:
        raise ValueError("--save is only valid in --train mode.")

    if not args.loadfile or not args.chessfile:
        raise ValueError("Missing LOADFILE or CHESSFILE.")

    if args.predict:
        network = load_network(args.loadfile)
        inputs = load_prediction_dataset(args.chessfile)
        inputs = inputs.reshape(inputs.shape[0], -1)

        outputs = network.forward(inputs)

        predicted_indices = np.argmax(outputs, axis=1)

        for idx in predicted_indices:
            print(get_label_from_index(idx))

    elif args.train:
        network = load_network(args.loadfile)
        inputs, labels = load_dataset(args.chessfile)
        inputs = inputs.reshape(inputs.shape[0], -1)

        (
            train_inputs,
            val_inputs,
            train_labels,
            val_labels,
        ) = train_validation_split(inputs, labels)

        optimizer = SGD(lr=0.01, weight_decay=0.001)

        history = train(
            network=network,
            optimizer=optimizer,
            train_inputs=train_inputs,
            train_labels=train_labels,
            val_inputs=val_inputs,
            val_labels=val_labels,
            loss_fn=cross_entropy_loss,
            loss_grad_fn=cross_entropy_grad,
            epochs=100,
            batch_size=32,
            early_stopping_patience=10,
            weight_decay=0.001,
        )

        final_train = history.train[-1] if history.train else None
        final_val = history.validation[-1] if history.validation else None
        num_epochs = len(history.train)

        print("\nTraining Summary:")
        if final_train:
            print(f"Final Training Loss: {final_train.loss:.4f}")
            print(f"Final Training Accuracy: {final_train.accuracy:.4f}")
        if final_val:
            print(f"Final Validation Loss: {final_val.loss:.4f}")
            print(f"Final Validation Accuracy: {final_val.accuracy:.4f}")
        print(f"Epochs Performed: {num_epochs}")

        if history.best_parameters is not None:
            for param, best_param in zip(network.parameters(), history.best_parameters):
                param[...] = best_param

        save_path = args.save if args.save else args.loadfile
        save_network(save_path, network)


def main() -> int:
    """
    Main entry point for the my_torch_analyzer CLI.
    """
    if "-h" in sys.argv or "--help" in sys.argv:
        print(HELP_TEXT)
        return EXIT_SUCCESS

    parser = SubjectArgumentParser(add_help=False)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save", type=str)
    parser.add_argument("loadfile", nargs="?")
    parser.add_argument("chessfile", nargs="?")

    try:
        args = parser.parse_args()
        run_analyzer(args)
        return EXIT_SUCCESS

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
