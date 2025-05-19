#!/home/czechen/Projects/Deep_Learning/NPFL/bin/python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset
from npfl138.trainable_module import TensorOrTensors

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, robeczech: transformers.PreTrainedModel,
                 dataset: ReadingComprehensionDataset.Dataset,backbone_freeze = True,tokenizer=None) -> None:
        super().__init__()
        self._tokenizer = tokenizer 
        # losses
        self._loss = torch.nn.BCELoss()

        self._freeze = backbone_freeze
        self._backbone = robeczech
        self._dropout = torch.nn.Dropout(0.1)

        self._output_start = torch.nn.Sequential(torch.nn.Linear(robeczech.config.hidden_size,1),
                                                 torch.nn.Sigmoid())
        self._output_end = torch.nn.Sequential(torch.nn.Linear(robeczech.config.hidden_size,1),
                                               torch.nn.Sigmoid())
    
    # TODO: Implement the model computation.
    def forward(self, input_ids: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
        if self._freeze:
            self._backbone.train(False)
            with torch.no_grad():
                result = self._backbone(input_ids,attention_mask=mask)
                hidden = result.last_hidden_state
        else:
            result = self._backbone(input_ids,attention_mask=mask)
            hidden = result.last_hidden_state
            hidden = self._dropout(hidden)
        start = self._output_start(hidden)
        end = self._output_end(hidden)
        return start,end

    def compute_loss(self, y_pred: TensorOrTensors, y: TensorOrTensors, *xs: tuple[torch.Tensor]) -> torch.Tensor:
        """Compute the loss of the model given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a sequence of tensors.
          y: The target output of the model, either a single tensor or a sequence of tensors.
          *xs: The inputs to the model, unpacked, if the input was a sequence of tensors.

        Returns:
          loss: The computed loss.
        """
        starts, ends = y_pred
        target_starts, target_ends = y 
        return self._loss(starts.squeeze(-1),target_starts) + self._loss(ends.squeeze(-1),target_ends)

    def predict_step(
        self, xs: TensorOrTensors, as_numpy: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """An overridable method performing a single prediction step.

        Parameters:
          xs: The input batch to the model, either a single tensor or a sequence of tensors.
          as_numpy: A flag controlling whether the output should be converted to a numpy array.

        Returns:
          predictions: The batch prediction.
        """
        input_ids, _ = xs
        with torch.no_grad():
            starts,ends = self(*xs)
        starts = torch.argmax(starts.squeeze(-1),dim=1)
        ends = torch.argmax(ends.squeeze(-1),dim=1)
        pred = [self._tokenizer.decode(list(input_ids[i,starts[i]:ends[i]+1])) for i in range(input_ids.shape[0])]
        return pred
             
class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: ReadingComprehensionDataset.Dataset, tokenizer=None,Test = False) -> None:

        super().__init__(dataset)
        self._tokenizer = tokenizer
        self._test = Test

    def transform(self, example):
        context = example['context']
        contexts = []
        questions = []
        starts = []
        ends = []
        
        if self._test:
            for qas in example['qas']:
                contexts.append(context)
                questions.append(qas['question'])
            return {'contexts': contexts,
                    'questions': questions}

        for qas in example['qas']:
            contexts.append(context)
            questions.append(qas['question'])
            starts.append(qas['answers'][0]['start'])
            ends.append(starts[-1]+len(qas['answers'][0]['text'])) #get the end index (in the context) of the answer 
        

        return {'contexts': contexts,
                'questions': questions,
                'starts': starts,
                'ends': ends}

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        #ques_conts,masks,starts, ends = zip(*batch)  
        #ques_conts_pad = torch.nn.utils.rnn.pad_sequence(ques_conts, batch_first=True)
        #print(ques_conts_pad.shape)
    
        # Initialize lists to collect all items
        batch_contexts = []
        batch_questions = []
        batch_starts = []
        batch_ends = []
       
        if self._test:
            for item in batch:
                batch_contexts.extend(item['contexts'])
                batch_questions.extend(item['questions'])


            tokenized = self._tokenizer(
                batch_contexts,
                batch_questions,
                truncation='only_first', 
                max_length=512,
                padding='longest',
                return_tensors='pt')

            inputs = (tokenized.input_ids,tokenized.attention_mask)
            return inputs


        # Collect items from each example in the batch
        for item in batch:
            batch_contexts.extend(item['contexts'])
            batch_questions.extend(item['questions'])
            batch_starts.extend(item['starts'])
            batch_ends.extend(item['ends'])

        tokenized = self._tokenizer(
                batch_contexts,
                batch_questions,
                truncation='only_first', 
                max_length=512,
                padding='longest',
                return_tensors='pt')
        
        inputs = (tokenized.input_ids,tokenized.attention_mask)
        batch_starts_tokenized = [tokenized.char_to_token(i,batch_starts[i]) for i in range(len(batch_starts))]
        batch_ends_tokenized = [tokenized.char_to_token(i,batch_ends[i]-1) for i in range(len(batch_ends))]
        largest_len = inputs[0].shape[1]
        outputs = (torch.nn.functional.one_hot(torch.tensor(batch_starts_tokenized),num_classes=largest_len).to(torch.float32),
                   torch.nn.functional.one_hot(torch.tensor(batch_ends_tokenized),num_classes=largest_len).to(torch.float32))
        return inputs,outputs


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

    # Load the pre-trained RobeCzech model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    
    # add special tokens
    tokenizer.add_special_tokens({'additional_special_tokens': [",",".",")","]","}"]})


    robeczech = transformers.AutoModel.from_pretrained("ufal/robeczech-base")
    # Load the data
    dataset = ReadingComprehensionDataset()
    
    # TODO: Prepare the data for training.
    train = TrainableDataset(dataset.train.paragraphs,tokenizer=tokenizer).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(dataset.dev.paragraphs, tokenizer=tokenizer).dataloader(batch_size=args.batch_size, shuffle=False)
    test = TrainableDataset(dataset.test.paragraphs, tokenizer=tokenizer,Test=True).dataloader(batch_size=args.batch_size, shuffle=False) 

    # Create the model.
    model = Model(args, robeczech, dataset.train,tokenizer = tokenizer)
    
    # TODO: Configure and train the model
    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={},
    )
    logs = model.fit(train, dev=dev, epochs=5)
    
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={'accuracy':torchmetrics.Accuracy('multiclass',num_classes=len(dataset.train.label_vocab))},
    )
    model._freeze = False

    logs = model.fit(train,dev,epochs=args.epochs)    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = model.predict(test,data_with_labels=True)
        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
