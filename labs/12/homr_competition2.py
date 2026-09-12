#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import torch
import torchvision.transforms.v2 as v2
import torchaudio.models.decoder

import npfl138
npfl138.require_version("2425.12")
from npfl138.datasets.homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN layer dimension.")
parser.add_argument("--res_blocks", default=4, type=int, help="Number of ResiudalBlocks to use")
parser.add_argument("--cnn_dim", default=32, type=int, help="CNN channels to use in ResiudalBlocks")

class ResiudalBlock(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self._direct_connection1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,2*in_channels,kernel_size=3,stride=2,padding=1,bias=False),
                torch.nn.BatchNorm2d(2*in_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2*in_channels,2*in_channels,kernel_size=3,stride=1,padding='same',bias=False),
                torch.nn.BatchNorm2d(2*in_channels))

        self._residual_connection = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,2*in_channels,kernel_size=3,stride=2,padding=1,bias=False),
                torch.nn.BatchNorm2d(2*in_channels))
        
        self._ReLU = torch.nn.ReLU()
        self._direct_connection2 = torch.nn.Sequential(
                torch.nn.Conv2d(2*in_channels,2*in_channels,kernel_size=3,stride=1,padding='same',bias=False),
                torch.nn.BatchNorm2d(2*in_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2*in_channels,2*in_channels,kernel_size=3,stride=1,padding='same',bias=False),
                torch.nn.BatchNorm2d(2*in_channels)) 

    def forward(self,inputs):
        direct1 = self._direct_connection1(inputs)
        residual1 = self._residual_connection(inputs)
        out1 = self._ReLU(direct1 + residual1)
        return self._ReLU(out1 + self._direct_connection2(out1))


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Define the model.
        self._args = args
        self._blank_token = 0
        self._CTCloss = torch.nn.CTCLoss(blank = 0,reduction='none')
        self._dropout = torch.nn.Dropout(args.dropout)
        
        in_channels = args.cnn_dim 
        self._initial_conv = torch.nn.Conv2d(HOMRDataset.C,in_channels,kernel_size=3,stride=1,padding='same')
        self._res_block = torch.nn.ModuleList()
        for _ in range(args.res_blocks):
            self._res_block.append(ResiudalBlock(in_channels))
            in_channels *= 2
        
        self._initial_LSTM_bidir = torch.nn.LSTM(input_size=5120, hidden_size= args.rnn_dim, batch_first=True,bidirectional=True)
        self._LSTM_1 = torch.nn.LSTM(args.rnn_dim,self._args.rnn_dim,batch_first=True,bidirectional=True)
        
        self._output_layer = torch.nn.Linear(args.rnn_dim,HOMRDataset.MARKS)
    
    def forward(self,images,images_lenghts) -> torch.Tensor:
        images = self._initial_conv(images)
        for resblock in self._res_block:
            images = resblock(images)
        images = torch.permute(images,(0,3,1,2))
        images = images.reshape(images.shape[0],images.shape[1],images.shape[2]*images.shape[3])
        seq_lengths = torch.round(images_lenghts / 16).to(torch.int)
          
        # Pack the sequence
        packed_images = torch.nn.utils.rnn.pack_padded_sequence(
            images, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
    
        # Process with LSTMs
        packed_first, _ = self._initial_LSTM_bidir(packed_images)
        # Unpack to add residual connections
        first, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_first, batch_first=True)
        first = first[..., 0:self._args.rnn_dim] + first[..., self._args.rnn_dim:]
        first_drop = self._dropout(first)

        # Pack again for second LSTM
        packed_first_drop = torch.nn.utils.rnn.pack_padded_sequence(
            first_drop, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_hidden1, _ = self._LSTM_1(packed_first_drop)
        hidden1, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_hidden1, batch_first=True)
 
        hidden1 = hidden1[..., 0:self._args.rnn_dim] + hidden1[..., self._args.rnn_dim:]
        hidden1 = self._dropout(hidden1) + first_drop
    
        return self._output_layer(hidden1)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor,*xs: tuple[torch.Tensor]) -> torch.Tensor:
        targets,target_lengths = y_true
        y_pred = torch.swapaxes(y_pred,0,1)
        _,input_lenghts = xs
        input_lenghts = torch.round(input_lenghts/16).to(torch.int)
        loss = self._CTCloss(y_pred,targets,input_lenghts,target_lengths)
        return torch.mean(loss)

    def ctc_decoding(self, y_pred: torch.Tensor, *xs: tuple[torch.Tensor]) -> list[torch.Tensor]:
        # Simple greedy decoding
        _, input_lengths = xs
        input_lengths = torch.round(input_lengths/16).to(torch.int)
        predictions = []
        for i in range(y_pred.shape[0]):
            # Get the most likely class at each timestep
            pred = torch.argmax(y_pred[i], dim=-1)
            
            # Truncate to actual length
            pred = pred[:input_lengths[i]]
            # Remove consecutive duplicates and blank tokens
            decoded = []
            prev_token = None
            for token in pred:
                if token != self._blank_token and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            predictions.append(decoded)
        return predictions

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *xs: tuple[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`. Consider computing it
        # only when `self.training==False` to speed up training.
#        if not self.training:
        predictions = self.ctc_decoding(y_pred,*xs)
        self.metrics["edit_distance"].update(predictions, y_true[0])
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            return batch

class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self,dataset,preprocessing):
        super().__init__(dataset)
        self._preprocessing = preprocessing

    def transform(self, example):
        image = example['image']
        marks = example['marks']
        image = self._preprocessing(image)
        return image,marks
            
    
    def collate(self, batch):
        # TODO: Construct a single batch from a list of individual examples.
        images_batch, marks_batch = zip(*batch)

        marks_lenghts = torch.tensor([len(x) for x in marks_batch])
        images_lenghts = torch.tensor([image.shape[-1] for image in images_batch])
        marks_batch = torch.nn.utils.rnn.pad_sequence(marks_batch,padding_value=0,batch_first=True)
        max_width = max(images_lenghts)
        padded_images = []

        for img in images_batch:
            pad_width = max_width - img.shape[-1]
            padded_img = torch.nn.functional.pad(img, (0, pad_width))  
            padded_images.append(padded_img)

        images_batch = torch.stack(padded_images)        
        return (images_batch,images_lenghts),(marks_batch,marks_lenghts)



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

    _preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Resize(160)
    ])
    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[1, HEIGHT, WIDTH]` tensor of `torch.uint8` values in [0-255] range,
    # - "marks", a `[num_marks]` tensor with indices of marks on the image.
    # Using `decode_on_demand=True` loads just the raw dataset (~500MB of undecoded PNG images)
    # and then decodes them on every access. Using `decode_on_demand=False` decodes the images
    # during loading, resulting in much faster access, but requires ~5GB of memory.
    homr = HOMRDataset(decode_on_demand=False)

    train = TrainableDataset(homr.train,preprocessing=_preprocessing).dataloader(args.batch_size, shuffle=True)
    dev = TrainableDataset(homr.dev,preprocessing=_preprocessing).dataloader(args.batch_size)
    test = TrainableDataset(homr.test,preprocessing=_preprocessing).dataloader(args.batch_size)

    # TODO: Create the model and train it
    model = Model(args)

    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0)

    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            metrics={"edit_distance": homr.EditDistanceMetric(ignore_index=0)},
            logdir=args.logdir
        )  
    model.to('cuda')
    logs = model.fit(train, dev=dev, epochs=args.epochs)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the sequences of recognized marks.
        predictions =  model.predict(test,data_with_labels=True)
        for sequence in predictions:
            print(" ".join(homr.MARK_NAMES[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
