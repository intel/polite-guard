# Polite Guard

Polite Guard is an open-source NLP language model developed by Intel, fine-tuned from BERT for text classification tasks. It is designed to classify text into four categories: polite, somewhat polite, neutral, and impolite. This model, along with its accompanying datasets and source code, is available on Hugging Face* and GitHub* to enable both communities to contribute to developing more sophisticated and context-aware AI systems. In this project, you will:
1. Generate your own synthetic data, and
2. Fine-tune a language model on your dataset using Intel technology.

## Table of Contents

- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model and Datasets](#model-and-datasets)
- [Articles](#articles)
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
1. **data-generator**: A package for generating synthetic datasets.
2. **fine-tuner**: A package for fine-tuning the BERT model for text classification utilizing Intel® Gaudi® Al accelerators.

## Getting Started

### Installation

Clone the repository:

```sh
git clone https://github.com/intel/polite-guard.git
```

Create and activate a virtual environment. Then install the dependencies for each package:

```sh
# For data-generator
cd polite-guard/data-generator
pip install -r requirements.txt

# For fine-tuner
cd ../fine-tuner
pip install -r requirements.txt
```

## Usage

### Data Generator

The `data-generator` package is responsible for generating and synthetic datasets with specified labels and categories. Refer to the [data-generator README](data-generator/README.md) for detailed instructions on how to run this synthetic data generator on the [Intel® Tiber™ AI Cloud](https://cloud.intel.com/).

### Fine-Tuner

The `fine-tuner` package is used to fine-tune the BERT model utilizing Intel® Gaudi® Al accelerators. Refer to the [fine-tuner README](fine-tuner/README.md) for detailed instructions on how to use this package.

## Model and Datasets

The Polite Guard model and datasets are available on Hugging Face. You can find them at the following links:

- [Polite Guard Model on Hugging Face](https://huggingface.co/Intel/polite-guard)
- [Polite Guard Datasets on Hugging Face](https://huggingface.co/datasets/Intel/polite-guard)

## Articles

To learn more about the implementation of the data generator and fine-tuner packages, refer to

- [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1), and 
- [How to Fine-Tune Language Models: First Principles to Scalable Performance](https://medium.com/p/78f42b02f112).

For more AI development how-to content, visit [Intel® AI Development Resources](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html).

## Join the Community
If you are interested in exploring other models, join us in the Intel and Hugging Face communities. These models simplify the development and adoption of Generative AI solutions, while fostering innovation among developers worldwide. Here are some ways you can contribute:

### 1. Star and Share
If you find this project valuable, please give it a star ★ on GitHub and share it with your network. Your support helps us grow the community and reach more contributors.

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