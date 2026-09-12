#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import torch
import torchaudio.models.decoder
import torchmetrics

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--rnn_dim", default=512, type=int, help="RNN layer dimension.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cuda", default=False,type=bool,help = "True when training on gpu" )

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()
        # TODO: Define the model.
        self._args = args
        self._blank_token = train.LETTER_NAMES[0]
        self._CTCloss = torch.nn.CTCLoss(blank = 0,reduction='none')
        if args.cuda:
            self._CTCDecoder = torchaudio.models.decoder.cuda_ctc_decoder(
                    tokens = train.LETTER_NAMES,
                    nbest=1,
                    beam_size=10,
                    blank_skip_threshold=0.95,
                    blank_id = 0)
        else:
            self._CTCDecoder = torchaudio.models.decoder.ctc_decoder(
                    lexicon = None,
                    tokens = train.LETTER_NAMES,
                    nbest=1,
                    beam_size=10,
                    blank_token=self._blank_token,
                    sil_token = self._blank_token,
                    )

        self._initial_LSTM_bidir = torch.nn.LSTM(input_size=13, hidden_size= args.rnn_dim, batch_first=True,bidirectional=True)
        self._dropout0 = torch.nn.Dropout(0.5)
        self._LSTM_1 = torch.nn.LSTM(args.rnn_dim,self._args.rnn_dim,batch_first=True,bidirectional=True)
        self._dropout1 = torch.nn.Dropout(0.5)

        self._LSTM_2 = torch.nn.LSTM(args.rnn_dim,self._args.rnn_dim,batch_first=True,bidirectional=True)
        self._dropout2 = torch.nn.Dropout(0.5)

        self._LSTM_3 = torch.nn.LSTM(args.rnn_dim,self._args.rnn_dim,batch_first=True,bidirectional=True)
        self._dropout3 = torch.nn.Dropout(0.5)

        self._LSTM_4 = torch.nn.LSTM(args.rnn_dim,self._args.rnn_dim,batch_first=True,bidirectional=True)
        self._dropout4 = torch.nn.Dropout(0.5)
        
        self._output_layer = torch.nn.Linear(args.rnn_dim,train.LETTERS)
    
    def forward(self,mfccs_sequence,mfccs_lenghts) -> torch.Tensor:
        first,_ = self._initial_LSTM_bidir(mfccs_sequence)
        first = first[...,0:self._args.rnn_dim] + first[...,self._args.rnn_dim:] 
        first_drop = self._dropout0(first)

        hidden1, _ = self._LSTM_1(first_drop)
        hidden1 = hidden1[...,0:self._args.rnn_dim] + hidden1[...,self._args.rnn_dim:] 
        hidden1 = self._dropout1(hidden1) + first_drop
 
        hidden2, _ = self._LSTM_2(first_drop)
        hidden2 = hidden2[...,0:self._args.rnn_dim] + hidden2[...,self._args.rnn_dim:] 
        hidden2 = self._dropout2(hidden2) + hidden1

        hidden3, _ = self._LSTM_3(first_drop)
        hidden3 = hidden3[...,0:self._args.rnn_dim] + hidden3[...,self._args.rnn_dim:] 
        hidden3 = self._dropout3(hidden3) + hidden2

        hidden4, _ = self._LSTM_4(first_drop)
        hidden4 = hidden4[...,0:self._args.rnn_dim] + hidden4[...,self._args.rnn_dim:] 
        hidden4 = self._dropout4(hidden4) + hidden3       
        out = self._output_layer(hidden4)
        return  torch.nn.functional.log_softmax(out,dim=2)
    
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,*xs: tuple[torch.Tensor]) -> torch.Tensor:
        # TODO: Compute the loss, most likely using the `torch.nn.CTCLoss` class.
        targets,target_lengths = y_true
        y_pred = torch.swapaxes(y_pred,0,1)
        _,input_lenghts = xs
        loss = self._CTCloss(y_pred,targets,input_lenghts,target_lengths)
        return torch.mean(loss)

    def ctc_decoding(self, y_pred: torch.Tensor, *xs: tuple[torch.Tensor]) -> list[torch.Tensor]:
        # TODO: Compute predictions, either using manual CTC decoding, or you can use:
        # - `torchaudio.models.decoder.ctc_decoder`, which is CPU-based decoding with
        #   rich functionality;
        #   - note that you need to provide `blank_token` and `sil_token` arguments
        #     and they must be valid tokens. For `blank_token`, you need to specify
        #     the token whose index corresponds to the blank token index;
        #     for `sil_token`, you can use also the blank token index (by default,
        #     `sil_token` has ho effect on the decoding apart from being added as the
        #     first and the last token of the predictions unless it is a blank token).
        # - `torchaudio.models.decoder.cuda_ctc_decoder`, which is faster GPU-based
        #   decoder with limited functionality.
        _,input_lengths = xs
        tokens_batch = []
        results = self._CTCDecoder(y_pred,input_lengths)
        for i in range(y_pred.shape[0]):
            tokens_batch.append(results[i][0].tokens)
        print(tokens_batch)
        return tokens_batch

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *xs: tuple[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
        predictions = self.ctc_decoding(y_pred,*xs)
        self.metrics["edit_distance"].update(predictions, y_true[0])
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self,dataset,vocab):
        super().__init__(dataset)
        self._letters_vocab = vocab

    def transform(self, example):
        # TODO: Prepare a single example. The structure of the inputs then has to be reflected
        # in the `forward`, `compute_loss`, and `compute_metrics` methods; right now, there are
        # just `...` instead of the input arguments in the definition of the mentioned methods.
        #
        # Note that while the `CommonVoiceCs.LETTER_NAMES` do not explicitly contain a blank token,
        # the [PAD] token can be employed as a blank token.
        mfccs = example['mfccs']
        sentence = example['sentence']
        sentence_char = list(sentence)
        sentence_indices = self._letters_vocab.indices(sentence_char)
        return mfccs,torch.tensor(sentence_indices)
            
    
    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        mfccs_batch, sentence_indices_batch = zip(*batch)
        mfccs_lenghts = torch.tensor([len(x) for x in mfccs_batch])
        sentence_lenghts = torch.tensor([len(x) for x in sentence_indices_batch])
        mfccs_batch = torch.nn.utils.rnn.pad_sequence(mfccs_batch,batch_first=True)
        sentence_indices_batch = torch.nn.utils.rnn.pad_sequence(sentence_indices_batch,batch_first=True)
        return (mfccs_batch,mfccs_lenghts),(sentence_indices_batch,sentence_lenghts)


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

    # Load the data.
    common_voice = CommonVoiceCs()

    train = TrainableDataset(common_voice.train,common_voice.letters_vocab).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(common_voice.dev,common_voice.letters_vocab).dataloader(args.batch_size)
    test = TrainableDataset(common_voice.test,common_voice.letters_vocab).dataloader(args.batch_size)
    
    # TODO: Create the model and train it. The `Model.compute_metrics` method assumes you
    # passed the following metric to the `configure` method under the name "edit_distance":
    #   CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)
    model = Model(args,common_voice)
    
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0)

    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            metrics={"edit_distance": common_voice.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)},
            logdir=args.logdir
        )  
    #logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test,data_with_labels=True)
        print(predictions)
        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTER_NAMES[char] for char in sentence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
