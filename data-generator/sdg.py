import argparse
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from huggingface_hub import login
from transformers import pipeline

import sdg_config


def read_token() -> None:
    """
    Reads a Hugging Face token from a file named 'token.txt' and logs in using the token.

    The file is expected to be located in the same directory as the script. If the file
    is missing, inaccessible, or another error occurs, the program will terminate with
    an appropriate error message.

    Raises:
        SystemExit: If the token file is not found, permission is denied, or any
                    other error occurs while reading the file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "token.txt")

    try:
        with open(path, "r") as file:
            token = file.read().strip()
    except FileNotFoundError:
        print(f"Error: The token file at {path} was not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied to read the token file at {path}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading token file at {path}: {e}")
        sys.exit(1)

    # Perform the login using the token
    login(token)


def parse_string(input_string: str) -> Tuple[str, str]:
    """
    Parses a string containing `OUTPUT:` and `REASONING:` sections and extracts their values.

    Args:
        input_string (str): The input string containing `OUTPUT:` and `REASONING:` labels.

    Returns:
        Tuple[str, str]: A tuple containing two strings:
                         - The content following `OUTPUT:`.
                         - The content following `REASONING:`.

    Raises:
        ValueError: If the input string does not match the expected format with both `OUTPUT:` and `REASONING:` sections.

    Note:
        - The function is case-sensitive and assumes `OUTPUT:` and `REASONING:` are correctly capitalized.
    """
    # Use regular expressions to extract OUTPUT and REASONING
    match = re.search(r"OUTPUT:\s*(.+?)\s*REASONING:\s*(.+)", input_string, re.DOTALL)

    if not match:
        raise ValueError(
            "The generated response is not in the expected 'OUTPUT:... REASONING:...' format."
        )

    # Extract the matched groups: output and reasoning
    output = match.group(1).strip()
    reasoning = match.group(2).strip()

    return output, reasoning


def sdg(
    sample_size: int,
    labels: List[str],
    label_descriptions: str,
    categories_types: Dict[str, str],
    use_case: str,
    prompt_examples: str,
    model: str,
    max_new_tokens: int,
    batch_size: int,
    output_dir: str,
    save_reasoning: bool,
) -> None:
    """
    Generates synthetic data based on specified categories and labels.

    Args:
        sample_size (int): The number of synthetic data samples to generate.
        labels (List[str]): The labels used to classify the synthetic data.
        label_descriptions (str): A description of the meaning of each label.
        categories_types (Dict[str, str]): The categories and their types for data generation and diversification.
        use_case (str): The use case of the synthetic data to provide context for the language model.
        prompt_examples (str): The examples used in the Few-Shot or Chain-of-Thought prompting.
        model (str): The large language model used for generating the synthetic data.
        max_new_tokens (int): The maximum number of new tokens to generate for each sample.
        batch_size (int): The number of samples per batch to append to the output file.
        output_dir (str): The directory path where the output file will be saved.
        save_reasoning (bool): Whether to save the reasoning or explanation behind the generated data.
    """

    categories = list(categories_types.keys())

    # Generate filename with current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

    # If sample_size is not divisible by batch_size, an extra batch is added
    num_batches = (sample_size + batch_size - 1) // batch_size

    print(
        f"\U0001F680  Synthetic data will be appended to {output_path} in {num_batches} batch(es)."
    )

    for batch in range(num_batches):
        # Calculate the start and end indices for the current batch
        start = batch * batch_size
        end = min(start + batch_size, sample_size)

        # Store results of the current batch
        batch_data = []

        # Assign random labels to the current batch
        batch_random_labels = random.choices(labels, k=batch_size)

        # Assign random categories to the current batch
        batch_random_categories = random.choices(categories, k=batch_size)

        for i in range(start, end):
            # Assign a random type to the ith category
            random_type = random.choices(
                categories_types[batch_random_categories[i - start]]
            )
            prompt = f"""You should create synthetic data for specified labels and categories. 
            This is especially useful for {use_case}.

            *Label Descriptions*
            {label_descriptions}

            *Examples*
            {prompt_examples}

            ####################

            Generate one output for the classification below.
            You may use the examples I have provided as a guide, but you cannot simply modify or rewrite them.
            Only return the OUTPUT and REASONING. 
            Do not return the LABEL, CATEGORY, or TYPE.

            LABEL: {batch_random_labels[i - start]}
            CATEGORY: {batch_random_categories[i - start]}
            TYPE: {random_type}
            OUTPUT:
            REASONING:
            """
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant designed to generate synthetic data for {use_case} with labels {labels} in categories {categories}.",
                },
                {"role": "user", "content": prompt},
            ]
            generator = pipeline("text-generation", model=model)
            result = generator(messages, max_new_tokens=max_new_tokens)[0][
                "generated_text"
            ][-1]["content"]

            # Uncomment to see the raw outputs
            # print(result)

            text, reasoning = parse_string(result)

            entry = {
                "text": text,
                "label": batch_random_labels[i - start],
                "model": model,
            }

            if save_reasoning:
                entry["reasoning"] = reasoning

            batch_data.append(entry)

        # Convert the batch results to a DataFrame
        batch_df = pd.DataFrame(batch_data)

        # Append the DataFrame to the CSV file
        if batch == 0:
            # If it's the first batch, write headers
            batch_df.to_csv(output_path, mode="w", index=False)
        else:
            # For subsequent batches, append without headers
            batch_df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"\U000026A1  Saved batch number {batch + 1}/{num_batches}")


def main() -> None:
    """
    Main entry point for running the synthetic data generator.

    This function performs the following steps:
    1. Reads the Hugging Face authentication token from the token file.
    2. Sets up and parses command-line arguments.
    3. Invokes the `sdg` function with the parsed arguments to generate synthetic data.

    Raises:
        SystemExit: If an error occurs during token reading or argument parsing.
    """

    read_token()

    parser = argparse.ArgumentParser(
        description="Run the synthetic data generator (sdg function)."
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="The number of samples generated by the language model (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="The language model for data generation (default: meta-llama/Meta-Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="The maximum number of new tokens to generate for each sample (default: 256)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="The batch size for saving generated samples to file (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="The output directory (default: ./)",
    )
    parser.add_argument(
        "--save_reasoning",
        action="store_true",
        help="Enable save reasoning (default: False)",
    )

    args = parser.parse_args()

    sdg(
        sample_size=args.sample_size,
        labels=sdg_config.labels,
        label_descriptions=sdg_config.label_descriptions,
        categories_types=sdg_config.categories_types,
        use_case=sdg_config.use_case,
        prompt_examples=sdg_config.prompt_examples,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_reasoning=args.save_reasoning,
    )


if __name__ == "__main__":
    main()
