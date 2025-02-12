# Fine-Tuning Language Models for Text Classification

With this package, you can fine-tune a pretrained language model for a text classification task using PyTorch Lightning*, Hugging Face* Transformers, and an accelerator of your choice. You may collect real labeled data for your task, or [generate synthetic data](../data-generator/). For more details, see [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1). The fine-tuner can be run on the [Intel® Tiber™ AI Cloud](https://cloud.intel.com/) environment. This platform provides ample computing resources ensuring smooth execution of your code.

## Table of Contents

- [Installation](#installation)
- [Preparation to Run on the Intel Tiber AI Cloud](#preparation-to-run-on-the-intel-tiber-ai-cloud)
- [Usage](#usage)
- [Logging and Checkpointing](#logging-and-checkpointing)
- [Functions and Classes](#functions-and-classes)
- [Article](#article)
- [Join the Community](#join-the-community)
- [License](#license)

## Installation

1. Clone the repository.

2. Create a virtual environment and activate it.

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```


## Preparation to Run on the Intel Tiber AI Cloud

1. Visit <https://cloud.intel.com/> and sign up.

2. Go to the "Learning" tab and click "Connect now" to launch JupyterLab*.

## Usage
To run the fine-tuner script, use a variation of the following command:

```sh
python fine-tune.py --train_data path/to/train.csv --val_data path/to/val.csv --test_data path/to/test.csv
```
### Arguments

- `--model_ckpt`: Pre-trained base model checkpoint (default: `bert-base-uncased`)
- `--num_labels`: Number of labels in the classification task (default: 4)
- `--train_data`: Path to the training CSV file (required)
- `--val_data`: Path to the validation CSV file (required)
- `--test_data`: Path to the test CSV file (required)
- `--batch_size`: Batch size for training and evaluation (default: 32)
- `--learning_rate`: Learning rate for the optimizer (default: 5e-5)
- `--weight_decay`: Weight decay for the optimizer (default: 0.01)
- `--max_epochs`: Number of epochs for training (default: 2)
- `--precision`: Precision for training (e.g., '16-mixed') (default: '16-mixed')
- `--patience`: Number of epochs with no improvement in the monitored metric after which training will be stopped (default: 3)
- `--min_delta`: Minimum change in the monitored metric to qualify as an improvement (default: 0.0)
- `--num_workers`: Number of worker threads for DataLoader (default: 6)
- `--accelerator`: Type of accelerator to use for training. Options include 'cpu', 'gpu', 'hpu', 'tpu', 'mps', and 'auto' (default: 'auto')
- `--devices`: Number of devices to use for training (default: 'auto')
- `--logger`: Logging framework to use. Options are 'tensorboard' or 'wandb' (default: 'tensorboard')
- `--log_dir`: Directory for saving logs (default: './logs')
- `--experiment_name`: Name of the experiment (default: None, will be auto-generated according to datetime and specified learning rate and batch size)

### Dataset Format

The input CSV files for training, validation, and testing should have the following format:

- `text`: The input text for classification.
- `label`: The numeric label for the classification task.

Example:

```csv
text,label
"This is a positive example.",1
"This is a negative example.",0
```
## Logging and Checkpointing

The script uses TensorBoard for logging and saves the best model checkpoint based on the validation F1 score. The logs and checkpoints are saved in the directory specified by the `--log_dir` argument.

### Example

Here is an example command to run the script:

```bash
python fine-tune.py \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --test_data data/test.csv \
    --model_ckpt bert-base-uncased \
    --num_labels 2 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --max_epochs 3 \
    --min_delta 0.005\
    --logger wandb\
    --log_dir ./logs \
    --experiment_name my_experiment
```

This command will fine-tune a BERT model on the specified training data, validate it on the validation data, and test it on the test data. The logs and checkpoints will be saved in the `./logs/my_experiment` directory.

## Functions and Classes

### `parse_devices(value: str) -> str | int`

Parses the devices argument for the number of devices to use for training. The argument can either be an integer, representing the number of devices, or the string 'auto', which automatically selects the available devices.

### `validate_positive_integer(value: str) -> int`

Validates that the input is a positive integer.

### `validate_positive_float(value: str) -> float`

Validates that the input is a positive float.

### `parse_args() -> argparse.Namespace`

Parses command-line arguments and returns them as an `argparse.Namespace` object.

### `load_csv(file_path: str) -> pd.DataFrame`

Loads a CSV file into a pandas DataFrame and performs checks to ensure it has the necessary columns and that the label column is numeric.

### `TextDataset`

A custom dataset class for text classification. It can accept either a pandas DataFrame or a list of strings. If a DataFrame is provided, it should have columns "text" (input text) and "label" (numeric labels). If a list of strings is provided, it will be used as the text data, and labels will be None.

### `prepare_data(train_path: str, val_path: str, test_path: str, tokenizer: PreTrainedTokenizer, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]`

Prepares data for training, validation, and testing by loading CSV files and creating corresponding datasets and dataloaders.

### `LightningModel`

A PyTorch Lightning model class for fine-tuning a language model on a classification task. It includes methods for training, validation, and testing steps, as well as configuring optimizers and logging metrics.

### `main() -> None`

The main function that trains and tests the model with user-specified arguments.

## Article
Visit [How to Fine-Tune Language Models: First Principles to Scalable Performance](https://medium.com/p/78f42b02f112) to learn more about the implementation of this package. For more AI development how-to content, visit [Intel® AI Development Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html).

## Join the Community
If you are interested in exploring other models, join us in the Intel and Hugging Face communities. These models simplify the development and adoption of Generative AI solutions, while fostering innovation among developers worldwide. Here are some ways you can contribute:

### 1. Star and Share
If you find this project valuable, please give it a star ★ on GitHub and share it with your network. Your support helps us grow the community and reach more contributors.

### 2. Contribute Code or Documentation
Help us improve and expand the project by contributing:
- **Code**: Fix bugs, optimize performance, or add new features.
- **Documentation**: Enhance the documentation to make it more accessible and user-friendly.

Check out the [Contributing Guide](../CONTRIBUTING.md) to get started.

### 3. Test and Provide Feedback
Run the software on your Intel hardware and share your experience. Report issues, suggest improvements, or request new features through the issues tab on GitHub.

### 4. Extend and Integrate
Use this project as a foundation for your own work. Build new applications or integrate it with other tools and libraries. Let us know what you create--we'd love to feature your work!

### 5. Spread the Word
Help us amplify our message by blogging, tweeting, or presenting about the project at conferences or meetups. Tag us and use our official hashtag so we can share your content with the community.

## License
This project is licensed under the [MIT License](../LICENSE).

*Other names and brands may be claimed as the property of others.
