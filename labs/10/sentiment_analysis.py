#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset,backbone_freeze = True) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self._freeze = backbone_freeze

        self._backbone = eleczech
        self._dropout = torch.nn.Dropout(0.1)
        self._output_layer = torch.nn.Linear(eleczech.config.hidden_size,len(dataset.label_vocab))

    # TODO: Implement the model computation.
    def forward(self, input_ids: torch.Tensor,mask: torch.tensor) -> torch.Tensor:
        if self._freeze:
            with torch.no_grad():
                result = self._backbone(input_ids,attention_mask=mask)
        else:
            result = self._backbone(input_ids,attention_mask=mask)
        hidden = result.last_hidden_state
        hidden = torch.sum(hidden,dim=1)
        hidden = self._dropout(hidden)
        output = self._output_layer(hidden)
        return output

class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset, tokenizer=None,Test = False) -> None:
        super().__init__(dataset)
        self._tokenizer = tokenizer
        self._test = Test

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        document = example['document']
        label = example['label']
        return document,label

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        documents,labels = zip(*batch)
        if not self._test:
            labels_ids = torch.tensor(self.dataset.label_vocab.indices(labels))
        else:
            labels_ids = torch.empty(len(batch))
        encoded_batch = self._tokenizer(documents,padding='longest')
        inputs = (torch.as_tensor(encoded_batch.input_ids),torch.as_tensor(encoded_batch.attention_mask))
        return (inputs,labels_ids) 


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train = TrainableDataset(facebook.train,tokenizer=tokenizer,).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(facebook.dev, tokenizer=tokenizer).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(facebook.test, tokenizer=tokenizer,Test=True).dataloader(batch_size=args.batch_size)

    # Create the model.
    model = Model(args, eleczech, facebook.train)

    # TODO: Configure and train the model
    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={'accuracy':torchmetrics.Accuracy('multiclass',num_classes=len(facebook.train.label_vocab))},
    )

    logs = model.fit(train, dev=dev, epochs=10)
    
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-5),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={'accuracy':torchmetrics.Accuracy('multiclass',num_classes=len(facebook.train.label_vocab))},
    )
    model._freeze = False

    logs = model.fit(train,dev,epochs=args.epochs)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test,data_with_labels=True)

        for document_logits in predictions:
            print(facebook.train.label_vocab.string(np.argmax(document_logits)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
