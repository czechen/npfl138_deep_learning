#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import torchvision
from torchvision.transforms import v2


import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--show_images", default=None, const=10, type=int, nargs="?", help="Show augmented images.")

class Resiudal(torch.nn.Module):
    def __init__(self,in_channels,double=False):
        super().__init__()
        self._double = double
        self.resiudal_block = torch.nn.Sequential(torch.nn.BatchNorm2d(in_channels),torch.nn.ReLU())
        if not double:
            self.resiudal_block.append(torch.nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1,padding='same',bias=False))
            self.resiudal_block.append(torch.nn.BatchNorm2d(in_channels//4))
            self.resiudal_block.append(torch.nn.ReLU())
            self.resiudal_block.append(torch.nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=1,padding='same',bias=False))
            self.resiudal_block.append(torch.nn.BatchNorm2d(in_channels//4))
            self.resiudal_block.append(torch.nn.ReLU())
            self.resiudal_block.append(torch.nn.Conv2d(in_channels//4,in_channels,1,1,padding='same',bias=False))
       
        else:
            self.resiudal_block_second_part = torch.nn.Sequential(torch.nn.Conv2d(in_channels,in_channels//2,kernel_size=1,stride=2,padding=1,bias=False),
                                                                  torch.nn.BatchNorm2d(in_channels//2),
                                                                  torch.nn.ReLU(),
                                                                  torch.nn.Conv2d(in_channels//2,in_channels//2,kernel_size=3,stride=1,padding='same',bias=False),
                                                                  torch.nn.BatchNorm2d(in_channels//2),
                                                                  torch.nn.ReLU(),
                                                                  torch.nn.Conv2d(in_channels//2,2*in_channels,1,1,padding='same',bias=False))

            self.residual_connection = torch.nn.Sequential(torch.nn.Conv2d(in_channels,2*in_channels,kernel_size=1,stride=2,padding=1,bias=False))

    def forward(self,inputs):
        if not self._double:
            layer_pass = self.resiudal_block(inputs)
            return inputs+layer_pass
        else:
            batch_relu = self.resiudal_block(inputs)
            res_conn = self.residual_connection(batch_relu)
            res_block = self.resiudal_block_second_part(batch_relu)
            return res_block + res_conn

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Here the `example` is already the selected example from the underlying
        # dataset; you now only need to process it as in the `ManualDataset`, so
        # (1) convert to `torch.float32`, (2) divide by 255, and (3) apply the
        # `self._augmentation_fn` if it is not `None`; finally, return (image, label) pair.
        image = example['image']
        label = example['label']
        image = image.to(torch.float32) / 255 
        if self._augmentation_fn != None:
            image = self._augmentation_fn(image)
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
    cifar = CIFAR10()
    augmentation_fn = v2.Compose([
            v2.RandomResize(28,36),
            v2.Pad(4),
            v2.RandomCrop(32),
            v2.RandomHorizontalFlip()
            ])
    train = TransformedDataset(cifar.train, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cifar.dev,augmentation_fn=None)
    test = TransformedDataset(cifar.test,augmentation_fn=None)

    train = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.dataloader_workers, persistent_workers=args.dataloader_workers > 0)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size, num_workers = args.dataloader_workers)
    # TODO: Create the model and train it.
    
    #model structure:
    #each integer indicates the number of repetition of the given block before doubling
    #the integers should be > 1
    structure = [4,5,5,3]
    num_of_channels = 64
    model = torch.nn.Sequential(torch.nn.Conv2d(3,num_of_channels,3,1,1))
    for index,value in enumerate(structure):
        if index == len(structure)-1:
            for i in range(value):
                model.append(Resiudal(num_of_channels,False))
        else:
            for i in range(value-1):
                model.append(Resiudal(num_of_channels,False))
            model.append(Resiudal(num_of_channels,True))
            num_of_channels *= 2

    model.append(torch.nn.AdaptiveAvgPool2d((1,1)))
    model.append(torch.nn.Flatten())
    model.append(torch.nn.Linear(num_of_channels,10))
    model = npfl138.TrainableModule(model)
    model.eval()(torch.zeros(1, CIFAR10.C, CIFAR10.H, CIFAR10.W))
    
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0001)
    
    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
            logdir=args.logdir
        )

    logs = model.fit(train,dev=dev,epochs=args.epochs, callbacks=[])
    

    if args.show_images:
        GRID, REPEATS, TAG = args.show_images, 5, "augmented" if args.augment else "original"
        for step in range(REPEATS):
            grid = torchvision.utils.make_grid([train[i][0] for i in range(GRID * GRID)], nrow=GRID)
            model.get_tb_writer("train").add_image(TAG, grid, step)
        print("Saved first {} training imaged to logs/{}".format(GRID * GRID, TAG))


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
