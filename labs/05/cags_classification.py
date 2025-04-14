#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchmetrics
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset,preprocessing=None, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Here the `example` is already the selected example from the underlying
        # dataset; you now only need to process it as in the `ManualDataset`, so
        # (1) convert to `torch.float32`, (2) divide by 255, and (3) apply the
        # `self._augmentation_fn` if it is not `None`; finally, return (image, label) pair.
        image = example['image']
        label = example['label']
        image = self._preprocessing(image)
        if self._augmentation_fn != None:
            image = self._augmentation_fn(image)
        return image, label


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace,backbone, backbone_freeze = True, _num_of_labels=34):
        super().__init__()
        self._args = args
        self._backbone = torch.nn.Sequential(backbone)
        self._num_of_labels = _num_of_labels
         
        self._backbone_freeze = backbone_freeze 
        if self._backbone_freeze:
            self._backbone.requires_grad_(False)
            self._backbone.eval()
        
        self._layers = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1536,self._num_of_labels),
                )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        with torch.no_grad():
            images_backbone = self._backbone(images) 
        return self._layers(images_backbone)

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

    # Load the ConvNextL model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1536]`.
    convnext_large = timm.create_model("convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
    
    # Create a simple preprocessing performing necessary normalization.
    
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=convnext_large.pretrained_cfg["mean"], std=convnext_large.pretrained_cfg["std"]),
    ])
    augmentation_fn = v2.Compose([
        v2.RandAugment()
        ])

    train = TransformedDataset(cags.train,preprocessing=preprocessing, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cags.dev,preprocessing=preprocessing,augmentation_fn=None)
    test = TransformedDataset(cags.test,preprocessing=preprocessing,augmentation_fn=None)
    
    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size, num_workers = args.dataloader_workers)
    
    # TODO: Create the model and train it.
    model = Model(args,backbone=convnext_large,_num_of_labels=cags.LABELS) 
   
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0001)
    
    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=0.1),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=cags.LABELS)},
            logdir=args.logdir
        )

    logs = model.fit(train,dev=dev,epochs=args.epochs, callbacks=[])

    #Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
