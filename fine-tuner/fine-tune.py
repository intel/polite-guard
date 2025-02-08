import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import F1Score
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


def parse_devices(value: str) -> str | int:
    """
    Parses the devices argument for the number of devices to use for training.

    The argument can either be an integer, representing the number of devices,
    or the string 'auto', which automatically selects the available devices.

    Args:
        value (str): The string representation of the argument value.

    Returns:
        str | int: Returns 'auto' if the value is 'auto', or an integer if
                  the value is a valid integer.

    Raises:
        argparse.ArgumentTypeError: If the value is neither 'auto' nor a valid integer.
    """
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device value: {value}. It must be an integer or 'auto'."
        )


def validate_positive_integer(value: str) -> int:
    """
    Validate that the input is a positive integer.

    Args:
        value: The input string from argparse

    Returns:
        int: The validated integer value

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError(
                f"The input value must be positive, got {int_value}"
            )
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")


def validate_positive_float(value: str) -> float:
    """
    Validate that the input is a positive float.

    Args:
        value: The input string from argparse

    Returns:
        float: The validated float value

    Raises:
        argparse.ArgumentTypeError: If validation fails
    """
    try:
        float_value = float(value)
        if float_value <= 0:
            raise argparse.ArgumentTypeError(
                f"The input value must be positive, got {float_value}"
            )
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {value}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for text classification."
    )

    # Model parameters
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained base model checkpoint",
    )
    parser.add_argument(
        "--num_labels",
        type=validate_positive_integer,
        default=4,
        help="Number of labels in classification task",
    )

    # Data parameters
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to the training CSV file"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to the validation CSV file"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to the test CSV file"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=validate_positive_integer,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=validate_positive_float,
        default=5e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=validate_positive_float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--max_epochs",
        type=validate_positive_integer,
        default=2,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--patience",
        type=validate_positive_integer,
        default=3,
        help="The number of epochs with no improvement in the monitored metric after which training will be stopped.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="The minimum change in the monitored metric to qualify as an improvement.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Precision for training (e.g., '16-mixed')",
    )
    parser.add_argument(
        "--num_workers",
        type=validate_positive_integer,
        default=6,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu", "hpu", "tpu", "mps", "auto"],
        default="auto",
        help="Type of accelerator to use for training. Options include 'cpu', 'gpu', 'hpu', 'tpu', 'mps', or 'auto' to automatically select the available hardware.",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default="auto",
        help="Number of devices to use for training. Can be an integer or 'auto' to automatically select the available devices.",
    )

    # Logging and checkpointing parameters
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory for saving logs"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment"
    )

    args = parser.parse_args()

    # Default experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-lr-{args.learning_rate}-bs-{args.batch_size}"

    return args


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame and performs checks to ensure
    it has the necessary columns and that the label column is numeric.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If the required columns ('text' and 'label') are missing.
        ValueError: If the 'label' column is not numeric.
        Exception: If any other error occurs when parsing the file.
    """
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(file_path)

        # Check if 'text' and 'label' columns exist
        if "text" not in df.columns or "label" not in df.columns:
            raise KeyError(
                f"The CSV file '{file_path}' must contain 'text' and 'label' columns."
            )

        # Check if the 'label' column is numeric
        if not pd.api.types.is_numeric_dtype(df["label"]):
            raise ValueError(
                f"The 'label' column in '{file_path}' must be numeric for fine-tuning."
            )

        return df

    except FileNotFoundError as e:
        print(f"Error: The file '{file_path}' was not found - {e}")
        raise
    except KeyError as e:
        print(f"Error: {e}")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error: There was an issue parsing the file '{file_path}' - {e}")
        raise


class TextDataset(Dataset):
    """
    Custom dataset for text classification. The dataset can accept either a pandas DataFrame or a list of strings.
    - If a DataFrame is provided, it should have columns "text" (input text) and "label" (numeric labels).
    - If a list of strings is provided, it will be used as the text data, and labels will be None.

    Args:
        data (Union[pd.DataFrame, List[str]]): The input data. If a DataFrame, it should have 'text' and 'label' columns.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text.
        max_length (int, optional): The maximum length for tokenized sequences. Default is 512.

    Attributes:
        data (Union[pd.DataFrame, List[str]]): The input data.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        max_length (int): The maximum length for tokenization.
        is_dataframe (bool): A flag indicating whether the input data is a DataFrame.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, List[str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize the class with data, tokenizer, and optional max_length.

        Args:
            data (Union[pd.DataFrame, List[str]]): The input data, either as a pandas DataFrame or a list of strings.
            tokenizer (PreTrainedTokenizer): The tokenizer used for processing text data.
            max_length (int, optional): The maximum length for tokenized sequences. Defaults to 512.

        Attributes:
            data: The input data, either a pandas DataFrame or a list of strings.
            tokenizer: The tokenizer for processing text.
            max_length: The maximum length for tokenized sequences.
            is_dataframe (bool): A flag indicating whether the data is a pandas DataFrame (True) or a list of strings (False).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_dataframe = hasattr(data, "iloc")  # Check if the data is a dataframe

    def __len__(self) -> int:
        """
        Return the number of elements in the data.

        Returns:
            int: The length of the dataframe or list.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Tokenize and return the input text and corresponding label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'attention_mask', and 'label' (if applicable).
        """
        if self.is_dataframe:
            row = self.data.iloc[idx]
            text = row["text"]
            label = row["label"]
        else:
            text = self.data[idx]
            label = None

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item


def prepare_data(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares data for training, validation, and testing by loading CSV files and creating corresponding datasets
    and dataloaders.

    Args:
        train_path (str): Path to the training CSV file.
        val_path (str): Path to the validation CSV file.
        test_path (str): Path to the testing CSV file.
        tokenizer (PreTrainedTokenizer): A tokenizer instance used for processing text data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the training, validation, and testing dataloaders.

    Raises:
        FileNotFoundError: If any of the CSV files do not exist.
        KeyError: If the CSV files are missing required columns ('text' and 'label').
        ValueError: If the 'label' column is not numeric.
        Exception: If any other error occurs when parsing the files or creating the datasets/dataloaders.
    """
    # Load CSV files
    train_df = load_csv(train_path)
    val_df = load_csv(val_path)
    test_df = load_csv(test_path)

    # Create datasets and dataloaders
    try:
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
        test_dataset = TextDataset(test_df, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
    except Exception as e:
        print(f"An error occurred in preparing dataloaders: {e}")
        raise
    return train_loader, val_loader, test_loader


class LightningModel(L.LightningModule):
    """
    PyTorch Lightning model class for fine-tuning a language model on a classification task.

    Args:
        model (AutoModelForSequenceClassification): The pre-trained base model.
        num_training_steps (int): The number of training samples times max_epochs for the learning rate scheduler.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        num_labels (int): The number of output labels for the classification task.
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        num_training_steps: int,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_labels: int = 4,
    ):
        """
        Initialize the model for sequence classification with hyperparameters and metrics.

        Args:
            model (AutoModelForSequenceClassification): The pre-trained model for sequence classification.
            num_training_steps (int): The total number of training steps.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 5e-5.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.01.
            num_labels (int, optional): The number of labels for classification. Defaults to 4.

        Attributes:
            model: The pre-trained model for sequence classification.
            num_training_steps: The total number of training steps.
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            num_labels: The number of labels for classification.
            f1_score: The F1 score metric for evaluation, weighted by class.
            val_f1: The F1 score metric for validation, weighted by class.
            test_f1: The F1 score metric for test, weighted by class.
            val_acc: The accuracy metric for validation.
            test_acc: The accuracy metric for test.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        self.f1_score = F1Score(
            num_classes=self.num_labels, task="multiclass", average="weighted"
        )
        self.val_f1 = F1Score(
            num_classes=self.num_labels, task="multiclass", average="weighted"
        )
        self.test_f1 = F1Score(
            num_classes=self.num_labels, task="multiclass", average="weighted"
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_labels
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor containing token IDs.
            attention_mask (torch.Tensor): Tensor indicating the attention mask for padding tokens.
            labels (Optional[torch.Tensor], optional): Labels for computing loss. Default is None.

        Returns:
            Dict: Output of the model.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def _shared_step(self, batch: Dict, stage: str) -> Optional[torch.Tensor]:
        """
        A shared step function for training, validation, and testing.

        Args:
            batch (Dict): A batch of data containing 'input_ids', 'attention_mask', and 'label'.
            stage (str): One of "train", "val", or "test" to indicate the current phase.

        Returns:
            Optional[torch.Tensor]: Returns the loss tensor for training; None for validation or testing.
        """
        outputs = self(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        logits = outputs["logits"]
        loss = outputs["loss"]
        labels = batch["label"]

        if stage == "train":
            self.log("train_loss", loss)
            return loss

        if stage == "val":
            self.val_acc(logits, labels)
            self.val_f1(logits, labels)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)

        if stage == "test":
            self.test_acc(logits, labels)
            self.test_f1(logits, labels)
            self.log("test_acc", self.test_acc, prog_bar=True)
            self.log("test_f1", self.test_f1, prog_bar=True)

        return None

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch (Dict): A batch of data, containing input tensors and labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value for this step that is sent to the optimizer.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """
        Perform a validation step.

        Args:
            batch (Dict): A batch of validation data.
            batch_idx (int): The index of the batch.
        """
        self._shared_step(batch, "val")

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """
        Perform a test step.

        Args:
            batch (Dict): A batch of test data.
            batch_idx (int): The index of the batch.
        """
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> Tuple[List[torch.optim.AdamW], List[Dict]]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[torch.optim.AdamW], List[Dict]]: A tuple containing a list of the optimizer(s)
            and a list of learning rate scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        num_warmup_steps = int(0.1 * self.num_training_steps)
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, self.num_training_steps
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def on_validation_epoch_start(self) -> None:
        """
        Resets the validation accuracy and F1 score metrics at the start of each validation epoch.
        """
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_start(self) -> None:
        """
        Resets the test accuracy and F1 score metrics at the start of each test epoch.
        """
        self.test_acc.reset()
        self.test_f1.reset()


def main() -> None:
    """
    Train and test the model with user-specified arguments.
    """

    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

    train_loader, val_loader, test_loader = prepare_data(
        args.train_data,
        args.val_data,
        args.test_data,
        tokenizer,
        args.batch_size,
        args.num_workers,
    )

    num_training_steps = len(train_loader) * args.max_epochs

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_ckpt, num_labels=args.num_labels
    )
    lightning_model = LightningModel(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        num_training_steps=num_training_steps,
    )

    # Setup callbacks and logger
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_f1"),
        EarlyStopping(
            monitor="val_f1",
            patience=args.patience,
            min_delta=args.min_delta,
            mode="max",
            verbose=True,
        ),
    ]
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator=args.accelerator,
        precision=args.precision,
        devices=args.devices,
        logger=logger,
        log_every_n_steps=10,
    )

    # Train the model
    try:
        trainer.fit(
            lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    except Exception as e:
        print(f"An error occurred in fine-tuning the model: {e}")
        raise

    # Test the best model
    try:
        trainer.test(lightning_model, test_loader, ckpt_path="best")
    except Exception as e:
        print(f"An error occurred in testing the fine-tuned model: {e}")
        raise


if __name__ == "__main__":
    main()
