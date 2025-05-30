#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import torchvision.transforms.v2 as v2
import torchmetrics
import torch

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.modelnet import ModelNet

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Resiudal(torch.nn.Module):
    def __init__(self,in_channels,double=False):
        super().__init__()
        self._double = double
        self.resiudal_block = torch.nn.Sequential(torch.nn.BatchNorm3d(in_channels),torch.nn.ReLU())
        if not double:
            self.resiudal_block.append(torch.nn.Conv3d(in_channels,in_channels//4,kernel_size=1,stride=1,padding='same',bias=False))
            self.resiudal_block.append(torch.nn.BatchNorm3d(in_channels//4))
            self.resiudal_block.append(torch.nn.ReLU())
            self.resiudal_block.append(torch.nn.Conv3d(in_channels//4,in_channels//4,kernel_size=3,stride=1,padding='same',bias=False))
            self.resiudal_block.append(torch.nn.BatchNorm3d(in_channels//4))
            self.resiudal_block.append(torch.nn.ReLU())
            self.resiudal_block.append(torch.nn.Conv3d(in_channels//4,in_channels,1,1,padding='same',bias=False))
       
        else:
            self.resiudal_block_second_part = torch.nn.Sequential(torch.nn.Conv3d(in_channels,in_channels//2,kernel_size=1,stride=2,padding=1,bias=False),
                                                                  torch.nn.BatchNorm3d(in_channels//2),
                                                                  torch.nn.ReLU(),
                                                                  torch.nn.Conv3d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding='same',bias=False),
                                                                  torch.nn.BatchNorm3d(in_channels//2),
                                                                  torch.nn.ReLU(),
                                                                  torch.nn.Conv3d(in_channels//2,2*in_channels,1,1,padding='same',bias=False))

            self.residual_connection = torch.nn.Sequential(torch.nn.Conv3d(in_channels,2*in_channels,kernel_size=1,stride=2,padding=1,bias=False))

    def forward(self,inputs):
        if not self._double:
            layer_pass = self.resiudal_block(inputs)
            return inputs+layer_pass
        else:
            batch_relu = self.resiudal_block(inputs)
            res_conn = self.residual_connection(batch_relu)
            res_block = self.resiudal_block_second_part(batch_relu)
            return res_block + res_conn


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace,num_of_labels=10):
        super().__init__()
        self._args = args
        self._num_of_labels = num_of_labels
        structure = [4,5,5,3]
        num_of_channels = 64
        self._layers = torch.nn.Sequential(torch.nn.Conv3d(1,num_of_channels,3,1,1))
        for index,value in enumerate(structure):
            if index == len(structure)-1:
                for i in range(value):
                    self._layers.append(Resiudal(num_of_channels,False))
            else:
                for i in range(value-1):
                    self._layers.append(Resiudal(num_of_channels,False))
                self._layers.append(Resiudal(num_of_channels,True))
                num_of_channels *= 2

        self._output_layer = torch.nn.Sequential(torch.nn.AdaptiveAvgPool3d((1,1,1)),
                                                 torch.nn.Flatten(),
                                                 torch.nn.Linear(num_of_channels,10))  
        
    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        hidden = self._layers(grids)
        output = self._output_layer(hidden)
        return output

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset,preprocessing=None, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing

    def transform(self, example: dict[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        grid = example['grid']
        label = example['label']
        image = self._preprocessing(grid)
        #if self._augmentation_fn != None:
        #    image = self._augmentation_fn(image)
        return image, label

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
    modelnet = ModelNet(args.modelnet)
    
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
    ])
    
    augmentation_fn = v2.Compose([
        v2.RandAugment()
        ])

    train = TransformedDataset(modelnet.train,preprocessing=preprocessing, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(modelnet.dev,preprocessing=preprocessing,augmentation_fn=None)
    test = TransformedDataset(modelnet.test,preprocessing=preprocessing,augmentation_fn=None)

    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size, num_workers = args.dataloader_workers)
    # TODO: Create the model and train it
    
    model = Model(args,num_of_labels=modelnet.LABELS) 
   
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0)
    
    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=modelnet.LABELS)},
            logdir=args.logdir
        )

    logs = model.fit(train,dev=dev,epochs=args.epochs, callbacks=[])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(grid, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
