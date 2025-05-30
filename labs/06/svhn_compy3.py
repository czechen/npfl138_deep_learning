#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision
import torchvision.transforms.v2 as v2
import torchmetrics
import bboxes_utils
import npfl138

from npfl138.trainable_module import is_sequence,validate_batch_input,validate_batch_input_output

from npfl138.trainable_module import TensorOrTensors
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--head_dim", default=256,type=int,help='Number of channels in classification/regression head') 
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--C4", default=None, type=str, help="Use only the C4 feature")
parser.add_argument("--scales", default=3, type=int, help="Number of scales to use")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")

TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3

def generate_anchors(p_level: int,scales: torch.Tensor):
    """
    Creates corresponding anchors (TOP,LEFT,BOTTOM,RIGHT) at each pixel of the pyramid level.

    args:
    p_level: pyramid level. Each square anchor (before scaling) will have size (2**(p_level))
    scales: scaling of the sides of anchors, list of tuples of type (vertical scaling, horizontal scaling)
    """
    anchors = []
    pyramid_level_lenght = 224//(2**p_level)
    for i in range(pyramid_level_lenght):
       for j in range(pyramid_level_lenght):
            for scale in scales:
                anchors.append([i*(2**p_level),j*(2**p_level),min(224, i*(2**p_level) + scale[0]*(2**p_level)),min(224, j*(2**p_level) + scale[1]*(2**p_level))])
    return torch.tensor(anchors) 

def resize_bboxes2(bboxes,original_image_shape,target_size=[[224,224]]):
    
    H,W = list(original_image_shape)
    target_size = list(target_size[0])
    bboxes_height,bboxes_width = bboxes[...,BOTTOM]-bboxes[...,TOP], bboxes[...,RIGHT]-bboxes[...,LEFT]
    bboxes_height_new = bboxes_height*(target_size[0]/H) 
    bboxes_width_new = bboxes_width*(target_size[1]/W)
    top_left_x,top_left_y = bboxes[...,1]*(target_size[1]/W),bboxes[...,0]*(target_size[0])/H

    return torch.stack([top_left_y,top_left_x,top_left_y+bboxes_height_new,top_left_x+bboxes_width_new],dim=1).to(torch.int)

def resize_bboxes(bboxes: torch.Tensor, original_image_shape: list[int] | tuple[int,int] | torch.Tensor, target_size: list[list[int]] | torch.Tensor = [[224,224]]):
    # Ensure original_image_shape is H, W
    if isinstance(original_image_shape, torch.Tensor):
        H_orig, W_orig = original_image_shape.tolist()
    else: # list or tuple
        H_orig, W_orig = original_image_shape[0], original_image_shape[1]

    # Ensure target_size is H, W
    if isinstance(target_size, torch.Tensor):
        H_target, W_target = target_size.tolist()
    elif isinstance(target_size[0], list) or isinstance(target_size[0], tuple): # nested list/tuple like [[H,W]]
         H_target, W_target = target_size[0]
    else: # flat list/tuple like [H,W]
         H_target, W_target = target_size[0], target_size[1]


    if H_orig == 0 or W_orig == 0: # Avoid division by zero for empty images or similar
        return torch.zeros_like(bboxes).to(torch.int)

    scale_y = H_target / H_orig
    scale_x = W_target / W_orig

    # bboxes are [..., TOP, LEFT, BOTTOM, RIGHT]
    # TOP = 0, LEFT = 1, BOTTOM = 2, RIGHT = 3
    new_top = bboxes[..., TOP] * scale_y
    new_left = bboxes[..., LEFT] * scale_x
    new_bottom = bboxes[..., BOTTOM] * scale_y
    new_right = bboxes[..., RIGHT] * scale_x
    
    # Stack them in the same order [TOP, LEFT, BOTTOM, RIGHT]
    # Ensure dim=1 for typical [N, 4] input, or dim=-1 if bboxes could have more leading dims
    return torch.stack([new_top, new_left, new_bottom, new_right], dim=-1).to(torch.int)   

class DevScore(npfl138.trainable_module.CallbackProtocol):
    def __init__(self,dev=None):
        super().__init__()
        self._dev = dev

    def __call__(self,module: npfl138.TrainableModule, epoch: int, logs) ->None:
        log_dir = {}
        for score in [0.2,0.25,0.3,0.35]:
                predictions = module.predict_test(self._dev,score)
                accuracy = SVHN.evaluate(getattr(SVHN(decode_on_demand=True), 'dev'),list(zip(*predictions)))
                log_dir[f'dev_score_{score}'] = accuracy
        module.log_config(log_dir)

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset,preprocessing=None, anchors = None,augmentation_fn=None,bbox_resize=None,label_smoothing=0.0,test=None) -> None:
        super().__init__(dataset)
        self._anchors = anchors
        self._augmentation_fn = augmentation_fn
        self._preprocessing = preprocessing
        self._bb_resize = bbox_resize
        self._label_smoothing = label_smoothing
        self._test = test

    def transform(self, example: dict[torch.tensor, torch.Tensor,torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example['image']
        original_image_shape = torch.tensor(image.shape[1:])
        image = self._preprocessing(image)
        classes = example['classes']
        bboxes = example['bboxes']
        if self._test:
            return image, torch.empty(1,10),torch.empty(1,1,4),original_image_shape #for test examples return only image and original_image_shape
        if self._label_smoothing:
            pass
        if self._augmentation_fn != None:
            image = self._augmentation_fn(image,bboxes)
        
        bboxes = resize_bboxes(bboxes,original_image_shape)
        classes,bboxes = bboxes_utils.bboxes_training(self._anchors,classes,bboxes,iou_threshold=0.5) #generate training examples
        classes_one_hot = torch.zeros((len(classes),10))

        #one hot representation, where background corresponds to all zeros

        for index,class_label in enumerate(classes):
            if class_label == 0:
                continue
            else:
                classes_one_hot[index][class_label-1] = 1
        return image, classes_one_hot, bboxes,original_image_shape

    def collate(self,batch):
        b_images,b_classes,b_bboxes,b_original_image_shape = zip(*batch)
        inputs = torch.stack(b_images,dim=0)
        targets = (torch.stack(b_classes,dim=0),torch.stack(b_bboxes,dim=0),torch.stack(b_original_image_shape,dim=0))
        return inputs, targets

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace,backbone,anchors: torch.Tensor, backbone_freeze = True, _num_of_labels=10):
        super().__init__()
        self._args = args
        self._backbone = backbone
        self._num_of_labels = _num_of_labels
        self._anchors = anchors
        self._num_of_anchors = len(anchors)//(14*14)
        self._upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self._train_iou_threshold = 0.5
        self._hubber_loss = torch.nn.HuberLoss(reduction='none')

        self._backbone.requires_grad_(False)
        self._backbone.eval()
        
        #process the C5 feature
        self._P5 = torch.nn.Sequential(
                torch.nn.Conv2d(192,args.head_dim,1,1,padding='same',bias=False))

        #process the C4 feature
        self._P4 = torch.nn.Sequential(
                torch.nn.Conv2d(112,args.head_dim,1,1,padding='same',bias=False))

        self._classification_head_C4 = torch.nn.Sequential(
                torch.nn.Conv2d(112,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,self._num_of_labels*self._num_of_anchors,3,1,padding='same'),
                )

        self._regression_head_C4 = torch.nn.Sequential(
                torch.nn.Conv2d(112,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,4*self._num_of_anchors,3,1,padding='same')
                )

        self._classification_head = torch.nn.Sequential(
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,self._num_of_labels*self._num_of_anchors,3,1,padding='same'),
                )

        self._regression_head = torch.nn.Sequential(
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,args.head_dim,3,1,padding='same',bias=False),
                torch.nn.BatchNorm2d(args.head_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(args.head_dim,4*self._num_of_anchors,3,1,padding='same')
                )
    def train(self, mode: bool = True):
        super().train(mode)
        self._backbone.train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        #images = images.to('cuda')
        with torch.no_grad():
            output,features = self._backbone.forward_intermediates(images)
            C4 = features[-2]
            C5 = features[-1]
    
        #Process the backbone features
        if self._args.C4 == 'P4':
            P4 = self._P4(C4)
            classification = self._classification_head(P4)
            regression = self._regression_head(P4)
            return classification,regression
        elif self._args.C4 == 'C4':
            classification = self._classification_head_C4(C4)
            regression = self._regression_head_C4(C4)
            return classification,regression
        P5 = self._P5(C5)
        P4 = self._P4(C4)

        P5_large = self._upsample(P5)

        P4_P5 = P4 + P5_large

        classification = self._classification_head(P4_P5)
        regression = self._regression_head(P4_P5)
        return classification,regression
    
    def train_step(self, xs: torch.Tensor, y: torch.Tensor):
        """An overridable method performing a single training step, returning the logs.

        Parameters:
          xs: The input batch to the model, either a single tensor or a sequence of tensors.
          y: The target output batch of the model, either a single tensor or a sequence of tensors.

        Returns:
          logs: A dictionary of logs from the training step
          """
        
        # unwrap the targets (anchors) and generate predictions
        anchor_classes,anchor_bboxes,_ = y 
        classes_pred,bboxes_pred = self(*xs)

        self.optimizer.zero_grad()
        
        losses = self.compute_loss(anchor_classes,anchor_bboxes,classes_pred,bboxes_pred)
        # Sum the classification (focal) loss with the regression (Huber) loss
        loss = losses[0]+losses[1]
        loss.backward()

        with torch.no_grad():
            self.optimizer.step()
            self.scheduler is not None and self.scheduler.step()
            return {"loss": self.loss_tracker(loss)} \
                | {"class_loss": losses[0]} \
                | {"regression_loss": losses[1]} \
                | {"total_loss": loss} \
                | ({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {})


    def compute_loss(self, anchor_classes: TensorOrTensors,
                     anchor_bboxes: TensorOrTensors,
                     classes_pred: TensorOrTensors, 
                     bboxes_pred: TensorOrTensors,
                     score: torch.float16=0.45) -> torch.Tensor:
       
        #Compute the loss of the model given the inputs, predictions, and target outputs.
        #reshape the predictions
        # Classes: (batch_size, 14,14, num_of_anchors (3), num of labels (10))
        # Bboxes: (batch_size, 14, 14, num_of_anchors (3), 4)

        classes_pred = classes_pred.reshape(-1,self._num_of_anchors,self._num_of_labels,14,14).permute(0,3,4,1,2)
        bboxes_pred = bboxes_pred.reshape(-1,self._num_of_anchors,4,14,14).permute(0,3,4,1,2)

        
        #reshape the targets (anchors) to match the predictions
        anchor_classes = anchor_classes.reshape(-1,14,14,self._num_of_anchors,self._num_of_labels)
        anchor_bboxes = anchor_bboxes.reshape(-1,14,14,self._num_of_anchors,4)
       
        #compute the classification loss
        class_loss = torchvision.ops.sigmoid_focal_loss(classes_pred,anchor_classes,alpha=0.5,gamma=2,reduction='sum') 

        # get the mask of the non-background targets, compute the huber loss for all examples
        # finally mask out background examples (so the loss is computed only for targets with non-background class)

        gt_positive_mask = (anchor_classes.sum(dim=-1) > 0).float()
        regression_loss = self._hubber_loss(bboxes_pred,anchor_bboxes) 
        regression_loss_per_anchor = regression_loss.sum(dim=-1)
        regression_loss = gt_positive_mask*regression_loss_per_anchor
        
        #compute the number of the non-background classes
        num_positive_anchors = gt_positive_mask.sum().clamp(min=1.0)

        # average the losses
        class_loss_norm = class_loss/num_positive_anchors
        regression_loss_norm = regression_loss.sum()/num_positive_anchors

        return (class_loss_norm,regression_loss_norm)

    def predict_test_step(self,xs,ys,score):
        with torch.no_grad():
            classes_pred,bboxes_pred = self(*xs)
        
        original_image_shape = list(ys[-1])
        
        #reshape the predictions and compute the probabilities
        classes_logits = classes_pred.view(-1,self._num_of_anchors,self._num_of_labels,14,14).permute(0,3,4,1,2).reshape(-1,self._num_of_labels)
        bboxes_deltas = bboxes_pred.view(-1,self._num_of_anchors,4,14,14).permute(0,3,4,1,2).reshape(-1, 4)
        class_probs = torch.sigmoid(classes_logits)
       
        # Convert the bboxes form relative (to anchors) representation to XYXY representation 
        bboxes = bboxes_utils.bboxes_from_rcnn(self._anchors.to('cpu'),bboxes_deltas.to('cpu'))
        bboxes = bboxes.to('cuda')
        final_boxes_list = []
        final_labels_list = []

        # perfrom the non-maximum suppression on non-background predictions
        for class_idx in range(self._num_of_labels):
            current_class_scores = class_probs[:, class_idx]

            score_mask = current_class_scores > score 
            if not score_mask.any(): 
                continue 


            selected_scores = current_class_scores[score_mask]
            selected_boxes = bboxes[score_mask]

            # Apply Non-Maximum Suppression (NMS)
            keep_indices = torchvision.ops.nms(
                selected_boxes,        
                selected_scores,        
                iou_threshold = 0.5 
            )
            
            # append the kept bboxes resized to the original image shape
            final_boxes_list.append(resize_bboxes(selected_boxes[keep_indices],(224,224),original_image_shape))
            
            final_labels_list.append(torch.full_like(selected_scores[keep_indices], class_idx, dtype=torch.long))
            
        
        if not final_boxes_list: # No detections after NMS across all classes
            return [-1], [[0,0,0,0]] #dummy value


        all_boxes_xyxy = torch.cat(final_boxes_list, dim=0)
        all_labels = torch.cat(final_labels_list, dim=0)
        return_labels = all_labels.cpu().tolist()
        return_boxes = all_boxes_xyxy.cpu().tolist()
        
        return return_labels, return_boxes
    
    def test_step(self, xs: TensorOrTensors, y: TensorOrTensors):
       with torch.no_grad():
            anchor_classes,anchor_bboxes,_ = y 
            classes_pred,bboxes_pred = self(*xs)
            losses = self.compute_loss(anchor_classes,anchor_bboxes,classes_pred,bboxes_pred)
            loss = losses[0]+losses[1]
            return {"loss": self.loss_tracker(loss)} \
                | {"class_loss": losses[0]} \
                | {"regression_loss": losses[1]} \
                | {"total_loss": loss} 
    
    def predict_test(self,dataloader: torch.utils.data.DataLoader,score):
        assert self.device is not None, "No device has been set for the TrainableModule, run configure first."
        self.eval()
        predicted_labels = []
        predicted_boxes = []
        for batch in dataloader:
            xs, y = validate_batch_input_output(batch)
            xs = tuple(x.to(self.device) for x in (xs if is_sequence(xs) else (xs,)))
            y = tuple(y_.to(self.device) for y_ in y) if is_sequence(y) else y.to(self.device) 
            labels,boxes = self.predict_test_step(xs,y,score)
            predicted_labels.append(labels)
            predicted_boxes.append(boxes)
        return predicted_labels,predicted_boxes


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

    # Load the ConvNextL model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = convnext_large.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1536, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution C2: 56x56, 56x56, C3: 28x28, C4: 14x14, C5: 7x7.
    '''
        torch.Size([1, 192, 56, 56])
        torch.Size([1, 192, 56, 56])
        torch.Size([1, 384, 28, 28])
        torch.Size([1, 768, 14, 14])
        torch.Size([1, 1536, 7, 7]  )
    '''
    efficient_net =  timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)
    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)
    if args.scales==1:
        anchors = generate_anchors(4,torch.tensor([(1,1)]))
    else:
        anchors = generate_anchors(4,torch.tensor([(2,1),(1,1),(1,2)]))

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.Resize(224),
        v2.ToDtype(torch.float32, scale=True),  # the `scale=true` also rescales the image to [0, 1].
        v2.Normalize(mean=efficient_net.pretrained_cfg["mean"], std=efficient_net.pretrained_cfg["std"]),
    ])

    bbox_resize = v2.Compose([
            v2.ToImage(),
            v2.Resize(224)
            ])
        
    train = TransformedDataset(svhn.train,preprocessing=preprocessing,bbox_resize=bbox_resize,anchors=anchors)
    dev = TransformedDataset(svhn.dev,preprocessing=preprocessing,augmentation_fn=None,bbox_resize=bbox_resize,anchors=anchors)
    test = TransformedDataset(svhn.test,preprocessing=preprocessing,augmentation_fn=None,bbox_resize=bbox_resize,anchors=anchors,test=True)

    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers,shuffle=True)
    dev_batch = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    dev_1 = dev.dataloader(batch_size=1, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=1, num_workers = args.dataloader_workers)        


    # TODO: Create the model and train it.
    model = Model(args,efficient_net,anchors)
    
    _optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer,T_max=args.epochs*len(train),eta_min = 0.0)

    model.configure(
            optimizer=_optimizer,
            scheduler=_scheduler,
            logdir=args.logdir
        )  
    logs = model.fit(train,dev=dev_batch,epochs=args.epochs, callbacks=[DevScore(dev=dev_1)])
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    for score in [0.25,0.27,0.3,0.35]:
        with open(os.path.join(args.logdir, f"svhn_competition_{score}.txt"), "w", encoding="utf-8") as predictions_file:
            # TODO: Predict the digits and their bounding boxes on the test set.
            # Assume that for a single test image we get
            # - `predicted_classes`: a 1D array with the predicted digits,
            # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
            predicted_classes_all, predicted_bboxes_all = model.predict_test(test,score)
            for predicted_classes, predicted_bboxes in zip(predicted_classes_all,predicted_bboxes_all):
                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    output += [int(label)] + list(map(float, bbox))
                print(*output, file=predictions_file)
    

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

