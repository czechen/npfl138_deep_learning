#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=4096, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=5, type=int, help="Window size to use.")
parser.add_argument("--embedding_dim",default=3,type=int,help='Length of embedded vectors')

class BatchGenerator:
    """A simple batch generator, optionally with shuffling.

    The functionality of this batch generator is very similar to
        torch.utils.data.DataLoader(
            torch.utils.data.StackDataset(inputs, outputs),
            batch_size=batch_size, shuffle=shuffle,
        )
    but if the data is stored in a single tensor, it is much faster.
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs)) if self._shuffle else torch.arange(len(self._inputs))
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.

        hidden_layer_size = 300
        self.layers = torch.nn.Sequential(

                torch.nn.Flatten(),
                torch.nn.Linear((2*args.window+1)*args.alphabet_size,hidden_layer_size),
                torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_size,hidden_layer_size),
                torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_layer_size,1)
                ) 

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        windows_one_hot = torch.nn.functional.one_hot(windows,self._args.alphabet_size).to(torch.float32)
        return self.layers(windows_one_hot).reshape(-1)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    
    torch.set_num_threads(8)
    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. The default label dtype of torch.float32 is suitable for binary classification,
    # but you should change it to torch.int64 if you use 2-class classification (CrossEntropyLoss).
    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.float32)

    # Instead of using
    #   train = torch.utils.data.DataLoader(
    #     torch.utils.data.StackDataset(uppercase_data.train.windows, uppercase_data.train.labels),
    #     batch_size=args.batch_size, shuffle=True)
    # we use the BatchGenerator, which is about an order of magnitude faster.
    train = BatchGenerator(uppercase_data.train.windows, uppercase_data.train.labels, args.batch_size, shuffle=True)
    dev = BatchGenerator(uppercase_data.dev.windows, uppercase_data.dev.labels, args.batch_size, shuffle=False)
    test = BatchGenerator(uppercase_data.test.windows, uppercase_data.test.labels, args.batch_size, shuffle=False)
    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    model = Model(args)
    _optimizer = torch.optim.AdamW(model.parameters(),lr=0.01,weight_decay=0.01)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0001)
    
    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            loss=torch.nn.BCEWithLogitsLoss(pos_weight=None),
            metrics={"accuracy": torchmetrics.Accuracy("binary",threshold=0.5)},
            logdir=args.logdir
        )

    logs = model.fit(train,dev=dev,epochs=args.epochs, callbacks=[])
    
    # TODO: Generate correctly capitalized test set. Use `uppercase_data.test.text`
    # as input, capitalize suitable characters, and write the result to `predictions_file`
    # (which is by default `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    sigmoid = torch.nn.Sigmoid()
    #test_prediction = np.round(model(uppercase_data.test.windows))
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        with torch.no_grad():
            i=0
            for windows, outputs in test:
                test_prediction = torch.round(sigmoid(model(windows)))
                for character in range(0,test_prediction.shape[0]):
                    if test_prediction[character] == 1:
                        print(uppercase_data.test.text[i+character].upper(), file=predictions_file, end='')
                    else:
                        print(uppercase_data.test.text[i+character].lower(), file=predictions_file, end='')
                i += test_prediction.shape[0]
        # Get the test set predictions; if you modified the `test` dataloader or your model
        # does not process the dataset windows, you might need to adjust the following line.
        predictions = model.predict(test, data_with_labels=True)
        ...


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
