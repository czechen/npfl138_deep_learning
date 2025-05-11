#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10, type=int, help="Show results every given batch.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""
    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output
        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size,attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size,attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim,1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        # TODO: Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, you have to deal with the fact that the first projection has
        #   shape `[batch_size, input_sequence_len, attention_dim]`, while the second projection has
        #   shape `[batch_size, attention_dim]`. The best solution is capable of creating the sum
        #   directly without creating any intermediate tensor.
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax activation, generating `weights`.
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input word had.
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.
        projected_decoded = self._project_decoder_layer(states)
        project_sum = torch.einsum('bia,ba->bia',self._encoded_projected,projected_decoded)
        project = torch.tanh(project_sum)
        output = self._output_layer(project)
        weights = torch.softmax(output,dim=1)
        attention = torch.einsum('bia,bi->ba',self._encoded,torch.squeeze(weights,dim=-1))
        return self._cell(torch.cat((inputs,attention),dim=1),states)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._args = args
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(len(self._source_vocab),args.cle_dim)
        self._source_rnn = torch.nn.GRU(args.cle_dim, args.rnn_dim,bidirectional=True, batch_first=True)

        # TODO: Define
        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=args.rnn_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `args.rnn_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.
        self._target_rnn_cell = WithAttention(torch.nn.GRUCell(args.cle_dim + args.rnn_dim,args.rnn_dim),attention_dim = args.rnn_dim)

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(args.rnn_dim,len(self._target_vocab))


        if not args.tie_embeddings:
            # TODO(lemmatizer_noattn): Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab),args.cle_dim)
        else:
            assert args.cle_dim == args.rnn_dim, "When tying embeddings, cle_dim and rnn_dim must match."
            # TODO(lemmatizer_noattn): Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            self._target_embedding = lambda x: torch.nn.functional.embedding(x,self._target_output_layer.weight)*(args.rnn_dim**(1/2))

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(self, words: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Embed the inputs using `self._source_embedding`.
        words_embed = self._source_embedding(words)
        # TODO: Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. Newly, the result should be encoding of every sequence element,
        # summing results in the opposite directions.
        lengths = (words > 0).to(torch.int)
        lengths = torch.sum(lengths,dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(words_embed, lengths,batch_first=True,enforce_sorted=False)
        hidden,h_n = self._source_rnn(packed)
        hidden,lenghts = torch.nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True)
        hidden = hidden[...,0:self._args.rnn_dim] + hidden[...,self._args.rnn_dim:]
        return hidden

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.
        inputs = torch.nn.functional.pad(targets,(1,0), value=MorphoDataset.BOW)
        inputs = inputs[:,:-1]

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - repeatedly call the `self._target_rnn_cell` on the sequence of embedded
        #   inputs and the previous states, starting with state `encoded[:, 0]`,
        #   obtaining outputs for all target hidden states,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.
        inputs_embed = self._target_embedding(inputs)
        state = encoded[:,0]
        hidden = torch.empty((*inputs.shape,self._args.rnn_dim))
        for i in range(inputs.shape[1]):
            state = self._target_rnn_cell(inputs_embed[:,i],state)
            hidden[:,i] = state
        output = self._target_output_layer(hidden)
        return torch.swapaxes(output,1,2)

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = encoded.shape[0]

        # TODO(decoder_training): Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # TODO: Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded[:, 0]`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.ones((batch_size))*MorphoDataset.BOW
        states = encoded[:,0]
        results = []
        result_lengths = torch.ones((batch_size))*max_length
        while index < max_length and torch.any(result_lengths == max_length):
            # TODO(lemmatizer_noattn):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            inputs_embed = self._target_embedding(inputs.to(torch.int))
            hidden = self._target_rnn_cell(inputs_embed,states)
            states = hidden
            outputs = self._target_output_layer(hidden)
            predictions = torch.argmax(outputs,dim=1)

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # TODO(lemmatizer_noattn): Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if self.training:  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=-2)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, :y.shape[-1]]
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        self.metrics["accuracy"].update(torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1))
        return {name: metric.compute() for name, metric in self.metrics.items()}  # Return all metrics.

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if self._show_results_every_batch and self._batches % self._show_results_every_batch == 0:
            self.log_console("{}: {} -> {}".format(
                self._batches,
                "".join(self._source_vocab.strings(xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True))),
                "".join(self._target_vocab.strings(self.predict_step((xs[0][:1],))[0]))))

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch]
            return [lemma.numpy(force=True) for lemma in batch] if as_numpy else batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO(lemmatizer_noattn): Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        return example['words'], example['lemmas'] 

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        words, lemmas = zip(*batch)
        # TODO: The `words` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `self.dataset.words.char_vocab` vocabulary.
        # Then create a tensor by padding the words to the length of the longest one in the batch.
        words = [word for word_list in words for word in word_list]
        words = [torch.tensor(self.dataset.words.char_vocab.indices(list(word))) for word in words]
        words = torch.nn.utils.rnn.pad_sequence(words,batch_first=True,padding_value=self.dataset.words.char_vocab.PAD)
        # TODO: Process `lemmas` analogously to `words`, but use `self.dataset.lemmas.char_vocab`,
        # and additionally, append `MorphoDataset.EOW` to the end of each lemma.
        lemmas = [lemma for lemma_list in lemmas for lemma in lemma_list] 
        lemmas = [torch.tensor(self.dataset.lemmas.char_vocab.indices(list(lemma)) + [MorphoDataset.EOW]) for lemma in lemmas]
        lemmas = torch.nn.utils.rnn.pad_sequence(lemmas,batch_first=True,padding_value=self.dataset.lemmas.char_vocab.PAD)
        # TODO: Return a pair (inputs, targets), where
        # - the inputs are words during inference and (words, lemmas) pair during training;
        # - the targets are lemmas.
        inputs = (words,lemmas)
        if self._training:
            return inputs, lemmas
        else:
            return words,lemmas 

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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt",max_sentences=args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    train = TrainableDataset(morpho.train, training=True).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev, training=False).dataloader(batch_size=args.batch_size)
    test = TrainableDataset(morpho.test, training=False).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO: Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO: Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        # TODO: Create a `torchmetrics.MeanMetric()` metric, where we will manually
        # collect lemmatization accuracy.
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use a different output structure than in lemmatizer_noattn.
        predictions = iter(model.predict(test, data_with_labels=True))

        for sentence in morpho.test.words.strings:
            for word in sentence:
                lemma = next(predictions)
                print("".join(morpho.test.lemmas.char_vocab.strings(lemma)), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
