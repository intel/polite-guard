# THIS PROJECT IS ARCHIVED  
Intel will not provide or guarantee development of or support for this project, including but not limited to, maintenance, bug fixes, new releases or updates.
Patches to this project are no longer accepted by Intel.  
This project has been identified as having known security issues. 

# Polite Guard
![Image representing the project's theme](polite-guard.png)

[Polite Guard](https://huggingface.co/Intel/polite-guard) is an open-source NLP language model developed by Intel, fine-tuned from BERT for text classification tasks. It is designed to classify text into four categories: polite, somewhat polite, neutral, and impolite. This model, along with its [accompanying dataset](https://huggingface.co/datasets/Intel/polite-guard) and source code, is available on Hugging Face* and GitHub* to enable both communities to contribute to developing more sophisticated and context-aware AI systems. In this project, you will:
1. Generate your own synthetic data, and
2. Fine-tune a language model on your dataset using Intel technology.

## Table of Contents

- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Articles](#articles)
- [Model and Dataset](#model-and-dataset)
- [Usage](#usage)
- [Join the Community](#join-the-community)
- [License](#license)

## Key Contributions

Polite Guard provides a scalable model development pipeline and methodology, making it easier for developers to create and fine-tune their own models. Other contributions of the project include:

1. **Improved Robustness**:
Polite Guard enhances the resilience of systems by providing a defense mechanism against adversarial attacks. This ensures that the model can maintain its performance and reliability even when faced with potentially harmful inputs. 
2. **Benchmarking and Evaluation**:
The project introduces the first politeness benchmark, allowing developers to evaluate and compare the performance of their models in terms of politeness classification. This helps in setting a standard for future developments in this area. 
3. **Enhanced Customer Experience**:
By ensuring respectful and polite interactions on various platforms, Polite Guard can significantly boost customer satisfaction and loyalty. This is particularly beneficial for customer service applications where maintaining a positive tone is crucial. 

## Repository Structure

This repository consists of two main packages:
1. **data-generator**: A package for generating synthetic datasets, featuring a [companion app](https://huggingface.co/spaces/Intel/synthetic-data-generator).
2. **fine-tuner**: A package for fine-tuning language models for text classification utilizing Al accelerators.

## Articles

To learn more about the implementation of the data generator and fine-tuner packages, refer to

- [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1), and 
- [How to Fine-Tune Language Models: First Principles to Scalable Performance](https://medium.com/p/78f42b02f112).

For more AI development how-to content, visit [Intel® AI Development Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html).

## Model and Dataset

The Polite Guard model and dataset are available on Hugging Face. You can find them at the following links:

- [Polite Guard Model on Hugging Face](https://huggingface.co/Intel/polite-guard)
- [Polite Guard Dataset on Hugging Face](https://huggingface.co/datasets/Intel/polite-guard)

## Usage

### Data Generator

The `data-generator` package is responsible for generating and synthetic datasets with specified labels and categories. Refer to the [data-generator README](data-generator/README.md) for detailed instructions on how to run this synthetic data generator on an AI PC or [Intel® Tiber™ AI Cloud](https://cloud.intel.com/).

### Fine-Tuner

The `fine-tuner` package is used to fine-tune language models utilizing Al accelerators such as Intel® Gaudi®. Refer to the [fine-tuner README](fine-tuner/README.md) for detailed instructions on how to use this package.

### Model and Dataset

You can use Polite Guard directly with a pipeline for categorizing text into classes polite, somewhat polite, neutral, and impolite.
```python
from transformers import pipeline

classifier = pipeline("text-classification", "Intel/polite-guard")
texts = [
    "I sincerely apologize for the inconvenience you've experienced. Please allow me a moment to resolve this for you as quickly as possible.",
    "I understand this isn't ideal, but could we move forward with this solution?",
    "The product specifications are as follows.",
    "You must be new here; you clearly don't know what you're doing.",
]
outputs = classifier(texts)
for text, output in zip(texts, outputs):
    print(f'"{text}": {output["label"]}')

```
**Output**
```text
"I sincerely apologize for the inconvenience you've experienced. Please allow me a moment to resolve this for you as quickly as possible.": polite
"I understand this isn't ideal, but could we move forward with this solution?": somewhat polite
"The product specifications are as follows.": neutral
"You must be new here; you clearly don't know what you're doing.": impolite
```

The (train, validation, and test) dataset(s) can be downloaded as follows.
```python
from datasets import load_dataset

dataset = load_dataset("Intel/polite-guard", split="train")
```
## Join the Community

If you are interested in exploring other models, join us in the Intel and Hugging Face communities. These models simplify the development and adoption of Generative AI solutions, while fostering innovation among developers worldwide. Here are some ways you can contribute:

### 1. Star and Share

If you find this project valuable, please give it a ⭐ on GitHub and share it with your network. Your support helps us grow the community and reach more contributors.

### 2. Contribute Code or Documentation

Help us improve and expand the project by contributing:
- **Code**: Fix bugs, optimize performance, or add new features.
- **Documentation**: Enhance the documentation to make it more accessible and user-friendly.

Check out the [Contributing Guide](CONTRIBUTING.md) to get started.

### 3. Test and Provide Feedback

Run the software on your Intel hardware and share your experience. Report issues, suggest improvements, or request new features through the issues tab on GitHub.

### 4. Extend and Integrate

Use this project as a foundation for your own work. Build new applications or integrate it with other tools and libraries. Let us know what you create--we'd love to feature your work!

### 5. Spread the Word

Help us amplify our message by blogging, tweeting, or presenting about the project at conferences or meetups. Tag us and use our official hashtag so we can share your content with the community.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

*Other names and brands may be claimed as the property of others.
