# Synthetic Data Generation

This open-source package includes a synthetic data generator that uses a language model to create data based on specified categories and labels. By using this package, you can generate your own synthetic data for various use cases, such as training and testing machine learning models. The synthetic data generator can be run on the [Intel® Tiber™ AI Cloud](https://cloud.intel.com/) environment, which is equipped with an Intel® Xeon® CPU. This platform provides ample computing resources ensuring smooth execution of your code.

## Table of Contents

- [Installation](#installation)
- [Preparation to Run on the Intel Tiber AI Cloud](#preparation-to-run-on-the-intel-tiber-ai-cloud)
- [Usage](#usage)
- [Configuration](#configuration)
- [Functions](#functions)
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

4. Add your Hugging Face token to a file named `token.txt` in the project directory.

## Preparation to Run on the Intel Tiber AI Cloud

1. Visit <https://cloud.intel.com/> and sign up.

2. Go to the "Learning" tab and click "Connect now" to launch JupyterLab*.

## Usage

To run the synthetic data generator, use a variation of the following command:

```sh
python sdg.py --sample_size 100 --batch_size 20 --output_dir ./output --model meta-llama/Meta-Llama-3.1-8B-Instruct --save_reasoning
```

### Arguments

- `--sample_size`: The number of samples generated by the language model (default: 100).
- `--labels`: The labels used to classify the synthetic data.
- `--label_descriptions`: A description of the meaning of each label.
- `--categories_types`: The categories and their types for data generation and diversification.
- `--use_case`: The use case of the synthetic data to provide context for the language model.
- `--prompt_examples`: The examples used in the Few-Shot or Chain-of-Thought prompting.
- `--model`: The language model for data generation (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`).
- `--max_new_tokens`: The maximum number of new tokens to generate for each sample (default: 256).
- `--batch_size`: The batch size for saving generated samples to file (default: 20).
- `--output_dir`: The output directory (default: `./`).
- `--save_reasoning`: Enable save reasoning (default: False).

## Configuration

The configuration for labels, label descriptions, categories, and other parameters is stored in the `sdg_config.py` file. Make sure to update this file with your specific configuration.

Example `sdg_config.py`:

```python
labels = ["Label1", "Label2", "Label3"]
label_descriptions = "Description of labels"
categories_types = {
    "Category1": ["Type1", "Type2"],
    "Category2": ["Type3", "Type4"]
}
use_case = "Your use case"
prompt_examples = "Examples for the Few-Shot Chain-of-Thought prompt."
```

## Functions

### `read_token() -> None`

Reads a Hugging Face token from a file named 'token.txt' and logs in using the token. (See (#article) to learn how to create and access your Hugging Face token.)

The file is expected to be located in the same directory as the script. If the file is missing, inaccessible, or another error occurs, the program will terminate with an appropriate error message.

#### Raises:
- `SystemExit`: If the token file is not found, permission is denied, or any other error occurs while reading the file.

### `parse_string(input_string: str) -> tuple[str, str]`

Parses a string containing `OUTPUT:` and `REASONING:` sections and extracts their values.

### `sdg(sample_size: int, labels: List[str], label_descriptions: str, categories_types: Dict[str, str], use_case: str, prompt_examples: str, batch_size: int, output_dir: str, model: str, save_reasoning: bool) -> None`

Generates synthetic data based on specified categories and labels.

### `main()`
The main function that parses command-line arguments and runs the synthetic data generator.

## Article

Visit [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1) to learn more about the implementation of this package. For more AI development how-to content, visit [Intel® AI Development Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html).

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