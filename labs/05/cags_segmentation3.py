#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")

class Residual(torch.nn.Module):
    def __init__(self,in_channels,double=False):
        super().__init__()
        self._double = double
        self.resiudal_block = torch.nn.Sequential()
        self.resiudal_block.append(torch.nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1,padding='same',bias=False))
        self.resiudal_block.append(torch.nn.BatchNorm2d(in_channels//4))
        self.resiudal_block.append(torch.nn.ReLU())
        self.resiudal_block.append(torch.nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=1,padding='same',bias=False))
        self.resiudal_block.append(torch.nn.BatchNorm2d(in_channels//4))
        self.resiudal_block.append(torch.nn.ReLU())
        self.resiudal_block.append(torch.nn.Conv2d(in_channels//4,in_channels,1,1,padding='same'))
       
    def forward(self,inputs):
        layer_pass = self.resiudal_block(inputs)
        return inputs+layer_pass

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset,preprocessing=None, augmentation_fn=None,label_smoothing=0.0) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing
        self._label_smoothing = label_smoothing

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example['image']
        mask = example['mask']
        if self._label_smoothing:
            pass
        image = self._preprocessing(image)
        if self._augmentation_fn != None:
            image = self._augmentation_fn(image)
        return image, mask


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace,backbone, backbone_freeze = True, _num_of_labels=34):
        super().__init__()
        self._args = args
        self._backbone = backbone
        self._num_of_labels = _num_of_labels
         
        self._backbone_freeze = backbone_freeze 
        if self._backbone_freeze:
            self._backbone.requires_grad_(False)
            self._backbone.eval()
        
        self._transposed_convolutions = []
        self._incoming_convolutions = []
        self._outgoing_convolutions = []
        output_channels = 1536
        incoming_channels = 768
        for _ in range(3):
            self._transposed_convolutions.append(torch.nn.Sequential(
                torch.nn.ConvTranspose2d(output_channels,output_channels // 2, kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(output_channels//2),
                torch.nn.ReLU()))
            output_channels //= 2

            self._incoming_convolutions.append(torch.nn.Sequential(torch.nn.Conv2d(incoming_channels,incoming_channels,kernel_size=3,stride=1,padding='same'))) 
            
            self._outgoing_convolutions.append(torch.nn.Sequential(
                torch.nn.Conv2d(incoming_channels,incoming_channels,kernel_size=3,stride=1,padding='same'),
                torch.nn.BatchNorm2d(incoming_channels),
                torch.nn.ReLU(),
                Residual(incoming_channels),
                Residual(incoming_channels),
                ))
            incoming_channels //= 2

        self._last_incoming = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(output_channels,output_channels,kernel_size=3,stride=1,padding='same')))

        self._224_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(output_channels,output_channels//2,kernel_size=2,stride=2,padding=0),
            torch.nn.BatchNorm2d(output_channels//2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(output_channels//2,output_channels//4,kernel_size=2,stride=2,padding=0),
            torch.nn.BatchNorm2d(output_channels//4),
            torch.nn.ReLU(),
            )
        self._input_conv = torch.nn.Sequential(torch.nn.Conv2d(3,output_channels//4,kernel_size=3,stride=1,padding='same'))
        self._last_conv = torch.nn.Sequential(torch.nn.Conv2d(output_channels//4,output_channels//4,kernel_size=3,stride=1,padding='same'),
                                              torch.nn.BatchNorm2d(output_channels//4),
                                              torch.nn.ReLU(),
                                              torch.nn.Conv2d(output_channels//4,1,kernel_size=1,stride=1,padding='same'))
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor: 
        images = images.to('cuda')
        with torch.no_grad():
            output,features = self._backbone.forward_intermediates(images)
            features = list(reversed(features[0:4]))
        for i in range(len(features)-1):
            transposed_output = self._transposed_convolutions[i](output)
            incoming = self._incoming_convolutions[i](features[i])
            output_pre = transposed_output+incoming
            output = self._outgoing_convolutions[i](output_pre)
        
        output_224 = self._224_conv(output + self._last_incoming(features[-1]))
        mask = self._last_conv(output_224+self._input_conv(images))
        return self._sigmoid(mask)

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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the ConvNextL model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = convnext_large.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1536, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 56x56, 56x56, 28x28, 14x14, 7x7.
    convnext_large = timm.create_model("convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=convnext_large.pretrained_cfg["mean"], std=convnext_large.pretrained_cfg["std"]),
    ])

    augmentation_fn = v2.Compose([
         v2.ColorJitter(brightness=.1, hue=.1),
        ])

    train = TransformedDataset(cags.train,preprocessing=preprocessing, augmentation_fn=None)
    dev = TransformedDataset(cags.dev,preprocessing=preprocessing,augmentation_fn=None)
    test = TransformedDataset(cags.test,preprocessing=preprocessing,augmentation_fn=None)
    
    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size, num_workers = args.dataloader_workers)
    # TODO: Create the model and train it.
    model = Model(args,backbone=convnext_large,_num_of_labels=cags.LABELS) 
    
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0001)
    IoU_metric = cags.MaskIoUMetric(from_logits=False)
    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            loss=torch.nn.BCELoss(),
            metrics={"IoU":IoU_metric},
            logdir=args.logdir
        )   
    model = model.to('cuda')
    for module in model._transposed_convolutions:
        module.to('cuda')
    for module in model._incoming_convolutions:
        module.to('cuda')
    for module in model._outgoing_convolutions:
        module.to('cuda')
    logs = model.fit(train,dev=dev,epochs=args.epochs, callbacks=[])
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
