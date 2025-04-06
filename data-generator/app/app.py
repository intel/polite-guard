import argparse
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
from huggingface_hub import login
from transformers import pipeline
from datasets import load_dataset

import gradio as gr

from huggingface_hub import HfApi, HfFolder


def extract_label(input_string: str) -> Tuple[str, str]:
    """
    Extracts the label and its description from a given input string.

    Args:
        input_string (str): The input string containing a label and its description, separated by a colon.

    Returns:
        Tuple[str, str]

    Raises:
        ValueError: If the input string does not contain a colon.
    """
    if ":" not in input_string:
        raise ValueError(
            "Input string must contain a ':' separating the label and description."
        )
    parts = input_string.split(":", 1)
    return parts[0].strip(), parts[1].strip()


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

    output = match.group(1).strip()
    reasoning = match.group(2).strip()

    return output, reasoning


def sdg(
    sample_size: int,
    labels: List[str],
    label_descriptions: str,
    categories_types: Dict[str, List[str]],
    use_case: str,
    prompt_examples: str,
    model: str,
    max_new_tokens: int,
    batch_size: int,
    output_dir: str,
    save_reasoning: bool,
) -> Tuple[str, str, str]:
    """
    Generates synthetic data based on specified categories and labels.

    Args:
        sample_size (int): The number of synthetic data samples to generate.
        labels (List[str]): The labels used to classify the synthetic data.
        label_descriptions (str): A description of the meaning of each label.
        categories_types (Dict[str, List[str]]): The categories and their types for data generation and diversification.
        use_case (str): The use case of the synthetic data to provide context for the language model.
        prompt_examples (str): The examples used in the Few-Shot or Chain-of-Thought prompting.
        model (str): The large language model used for generating the synthetic data.
        max_new_tokens (int): The maximum number of new tokens to generate for each sample.
        batch_size (int): The number of samples per batch to append to the output file.
        output_dir (str): The directory path where the output file will be saved.
        save_reasoning (bool): Whether to save the reasoning or explanation behind the generated data.

    Returns:
        Tuple[str, str, str]: A tuple containing:
                              - A status message indicating the save location of the synthetic data.
                              - The path to the output CSV file.
                              - The timestamp used in the filename.
    """
    categories = list(categories_types.keys())

    # Generate filename with current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}.csv")

    num_batches = (sample_size + batch_size - 1) // batch_size

    for batch in range(num_batches):
        start = batch * batch_size
        end = min(start + batch_size, sample_size)

        batch_data = []

        batch_random_labels = random.choices(labels, k=batch_size)
        batch_random_categories = random.choices(categories, k=batch_size)

        for i in range(start, end):
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
            Only return the OUTPUT and REASONING. The first token in your response must be OUTPUT.
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
                    "content": f"You are a helpful assistant designed to generate synthetic data for {use_case} with labels {labels} in categories {categories}. The first token in your generated text must be OUTPUT: This must be followed by the token REASONING: as in the prompt examples.",
                },
                {"role": "user", "content": prompt},
            ]
            generator = pipeline("text-generation", model=model)
            result = generator(messages, max_new_tokens=max_new_tokens)[0][
                "generated_text"
            ][-1]["content"]

            text, reasoning = parse_string(result)

            entry = {
                "text": text,
                "label": batch_random_labels[i - start],
                "source": model,
            }

            if save_reasoning:
                entry["reasoning"] = reasoning

            batch_data.append(entry)

        batch_df = pd.DataFrame(batch_data)

        if batch == 0:
            batch_df.to_csv(output_path, mode="w", index=False)
        else:
            batch_df.to_csv(output_path, mode="a", header=False, index=False)

    return f"Synthetic data saved to {output_path}", output_path, timestamp


def main() -> None:
    """
    Main entry point for running the synthetic data generator.
    """

    def run_sdg(
        sample_size: int,
        model: str,
        max_new_tokens: int,
        save_reasoning: bool,
        token: str,
        state: Dict,
        label_boxes: List[Dict[str, str]],
        use_case: str,
        prompt_examples: str,
        category_boxes: List[Dict[str, str]],
    ) -> Tuple[str, Dict]:
        """
        Runs the synthetic data generation process and updates the application state.

        Args:
            sample_size (int): The total number of synthetic data samples to generate.
            model (str): The large language model used for generating the synthetic data.
            max_new_tokens (int): The maximum number of new tokens to generate for each sample.
            save_reasoning (bool): Whether to save the reasoning or explanation behind the generated data.
            token (str): The Hugging Face token for authentication.
            state (Dict): The application state to store the output path and timestamp.
            label_boxes (List[Dict[str, str]]): A list of label description dictionaries.
            use_case (str): A string for the use case description.
            prompt_examples (str): A string for prompt examples.
            category_boxes (List[Dict[str, str]]): A list of category and type dictionaries .

        Returns:
            Tuple[str, Dict]: A tuple containing:
                                  - A status message indicating the result of the generation process.
                                  - The updated application state with the output path and timestamp.
        """
        try:
            login(token)
        except Exception as e:
            return f"Error logging in with token: {e}", None, None

        label_descriptions = ""
        labels = []
        for box in label_boxes:
            label_descriptions += box["content"] + "\n"
            label, _ = extract_label(box["content"])
            labels.append(label)

        categories_types = {}
        for box in category_boxes:
            category, types = extract_label(box["content"])
            categories_types[category.strip()] = [t.strip() for t in types.split(",")]

        status, output_path, timestamp = sdg(
            sample_size=sample_size,
            labels=labels,
            label_descriptions=label_descriptions,
            categories_types=categories_types,
            use_case=use_case,
            prompt_examples=prompt_examples,
            model=model,
            max_new_tokens=max_new_tokens,
            batch_size=20,
            output_dir="./",
            save_reasoning=save_reasoning,
        )
        state["output_path"] = output_path
        state["timestamp"] = timestamp
        return status, state

    with gr.Blocks(css_paths="styles.css") as demo:
        gr.Markdown(
            "# Synthetic Data Generator",
            elem_id="header",
            elem_classes="text-center",
        )
        gr.Markdown(
            "**Use Language Models to Create Datasets for Specified Labels and Categories**",
            elem_classes="text-center",
        )
        with gr.Tab("Data Generator"):
            with gr.Row():  # A row for two columns
                with gr.Column():  # First column
                    gr.Markdown(
                        "## Setup & Configure",
                        elem_classes="text-center",
                    )
                    gr.Markdown("### Use Case")
                    use_case = gr.Textbox(
                        show_label=False,
                        placeholder="Describe your use case (e.g., customer service).",
                        autofocus=True,
                    )
                    label_boxes = gr.State([])
                    gr.Markdown(
                        "### Labels\nUse a colon to separate each label and its description as in 'label: description.'"
                    )
                    with gr.Row():
                        new_label = gr.Textbox(
                            show_label=False,
                        )
                        gr.Examples(
                            examples=[
                                "polite: Text is considerate and shows respect and good manners, often including courteous phrases and a friendly tone.",
                                "somewhat polite: Text is generally respectful but lacks warmth or formality, communicating with a decent level of courtesy.",
                            ],
                            example_labels=["polite", "somewhat polite"],
                            inputs=new_label,
                        )

                    add_label_button = gr.Button("Save Label", elem_classes="btn")

                    def add_item(
                        label_boxes: List[Dict[str, str]], new_content: str
                    ) -> Tuple[List[Dict[str, str]], str]:
                        """
                        Adds a new label or category to the list if the input is not empty.

                        Args:
                            label_boxes (List[Dict[str, str]]): A list containing dictionaries representing the current labels or categories.
                            new_content (str): The new label or category content to add.

                        Returns:
                            Tuple[List[Dict[str, str]], str]: A tuple containing the updated list of labels or categories and an empty string to clear the input field.
                        """
                        if new_content.strip():
                            return (
                                label_boxes + [{"content": new_content.strip()}],
                                "",
                            )
                        return label_boxes, ""

                    add_label_button.click(
                        add_item, [label_boxes, new_label], [label_boxes, new_label]
                    )

                    @gr.render(inputs=label_boxes)
                    def render_boxes(box_list: List[Dict[str, str]]) -> None:
                        """
                        Renders a list of labels in a Gradio interface.

                        Args:
                            box_list (List[Dict[str, str]]): A list containing dictionaries representing the categories to render.
                        """
                        with gr.Accordion(f"Saved Labels ({len(box_list)})"):
                            for box in box_list:
                                with gr.Row():
                                    gr.Textbox(
                                        box["content"],
                                        lines=2,
                                        show_label=False,
                                        container=False,
                                    )
                                    delete_button = gr.Button(
                                        "Delete", scale=0, variant="stop"
                                    )

                                def delete(
                                    box: Dict[str, str] = box,
                                ) -> List[Dict[str, str]]:
                                    """
                                    Deletes a specific box from the list of labels.

                                    Args:
                                        box (Dict[str, str]): The box to be removed from the list.

                                    Returns:
                                        List[Dict[str, str]]: The updated list of labels after the box is removed.
                                    """
                                    box_list.remove(box)
                                    return box_list

                                delete_button.click(delete, None, [label_boxes])

                    category_boxes = gr.State([])
                    gr.Markdown(
                        "### Categories\nUse a colon to separate each category and its subcategories as in 'category: type1, type2.'"
                    )
                    with gr.Row():
                        new_category = gr.Textbox(show_label=False)
                        gr.Examples(
                            examples=[
                                "travel: hotel, airline, train",
                                "finance: fees and charges, credit",
                            ],
                            example_labels=["travel", "finance"],
                            inputs=new_category,
                        )

                    add_category_button = gr.Button("Save Category", elem_classes="btn")
                    add_category_button.click(
                        add_item,
                        [category_boxes, new_category],
                        [category_boxes, new_category],
                    )

                    @gr.render(inputs=category_boxes)
                    def render_boxes(box_list: List[Dict[str, str]]) -> None:
                        """
                        Renders a list of categories in a Gradio interface.

                        Args:
                            box_list (List[Dict[str, str]]): A list containing dictionaries representing the categories to render.
                        """
                        with gr.Accordion(f"Saved Categories ({len(box_list)})"):
                            for box in box_list:
                                with gr.Row():
                                    gr.Textbox(
                                        box["content"],
                                        show_label=False,
                                        container=False,
                                    )
                                    delete_button = gr.Button(
                                        "Delete", scale=0, variant="stop"
                                    )

                                def delete(
                                    box: Dict[str, str] = box,
                                ) -> List[Dict[str, str]]:
                                    """
                                    Deletes a specific box from the list of categories.

                                    Args:
                                        box (Dict[str, str]): The box to be removed from the list.

                                    Returns:
                                        List[Dict[str, str]]: The updated list of categories after the box is removed.
                                    """
                                    box_list.remove(box)
                                    return box_list

                                delete_button.click(delete, None, [category_boxes])

                    gr.Markdown(
                        "### Guiding Examples\nInclude all examples in this box. For each example, provide a LABEL, CATEGORY, TYPE, OUTPUT, and REASONING."
                    )
                    with gr.Row():
                        prompt_examples = gr.Textbox(
                            show_label=False,
                        )
                        gr.Examples(
                            label="Example",
                            examples=[
                                """LABEL: polite
CATEGORY: food and drink
TYPE: cafe
OUTPUT: Thank you for visiting! While we prepare your coffee, feel free to relax or browse our selection of pastries. Let us know if we can make your day even better!
REASONING: This text is polite because it expresses gratitude and encourages the customer to feel at ease with a welcoming tone. Phrases like "Let us know if we can make your day even better" show warmth and consideration, enhancing the customer experience.

LABEL: somewhat polite
CATEGORY: travel
TYPE: train
OUTPUT: I understand your concern about your booking, and I'll check what options we have for you.
REASONING: This text would be classified as "somewhat polite." The acknowledgment of the customer's concern shows a basic level of respect. The sentence is direct and lacks additional warmth or formality, but it communicates a willingness to help. The use of "I'll check" is a straightforward commitment to action without additional courteous phrases that would make it fully polite.
"""
                            ],
                            example_labels=["polite and somewhat polite"],
                            inputs=prompt_examples,
                        )

                    gr.Markdown(
                        """### Language Model
                    Visit [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [Gemma 3](https://huggingface.co/google/gemma-3-1b-it), and submit an access request for these gated Hugging Face repositories."""
                    )
                    model = gr.Dropdown(
                        label="Model",
                        choices=[
                            "google/gemma-3-1b-it",
                            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                            "meta-llama/Llama-3.2-3B-Instruct",
                        ],
                        value="google/gemma-3-1b-it",
                    )
                    max_new_tokens = gr.Number(
                        label="Maximum Number of New Tokens", value=256, minimum=64
                    )
                    token = gr.Textbox(
                        label="Hugging Face Token",
                        type="password",
                        info="Enter a 'Read' Hugging Face token to access gated language models, or a 'Write' token to push the generated data to Hugging Face.",
                    )
                with gr.Column():  # Second column
                    gr.Markdown(
                        "## Generate & Export",
                        elem_classes="text-center",
                    )
                    gr.Markdown("### Status")
                    status = gr.Textbox(show_label=False)
                    gr.Markdown(
                        "### Actions\nEnter the number of rows for the generated dataset, and optionally a Hugging Face repo ID."
                    )
                    sample_size = gr.Number(label="Sample Size", value=1, minimum=1)
                    save_reasoning = gr.Checkbox(label="Save Reasoning", value=True)
                    repo_id = gr.Textbox(
                        label="Hugging Face Repo ID (Optional)",
                        placeholder="your-username/your-repo-name",
                    )
                    is_public_repo = gr.Checkbox(
                        label="Make Repository Public", value=False
                    )
                    with gr.Row():
                        generate_button = gr.Button(
                            "Generate", interactive=False, elem_classes="btn"
                        )
                        download_button = gr.Button(
                            "Download", interactive=False, elem_classes="btn"
                        )
                        push_button = gr.Button(
                            "Push to 🤗",
                            interactive=False,
                            variant="huggingface",
                        )
                    file_output = gr.File(label="Download", visible=False)
                    gr.Markdown(
                        "### Sample Output",
                    )
                    dataset = load_dataset(
                        "intel/polite-guard",
                        data_files={"validation": "data/validation/val_cot.csv"},
                    )
                    dataset = dataset["validation"].to_pandas().sample(n=5)

                    dataframe = gr.Dataframe(value=dataset, show_label=False)

                state = gr.State({"output_path": None, "timestamp": None})

            def toggle_button(
                token_value: str,
                label_value: List[Dict[str, str]],
                category_value: List[Dict[str, str]],
                use_case_value: str,
                example_value: str,
            ) -> Dict[str, Any]:
                """
                Toggles the interactivity of the generate button based on input values.

                Args:
                    token_value (str): The Hugging Face token value.
                    label_value (List[Dict[str, str]]): A list of label description dictionaries.
                    category_value (List[Dict[str, str]]): A list of category and type dictionaries.
                    use_case_value (str): A string for the use case description.
                    example_value (str): A string for prompt examples.

                Returns:
                    Dict[str, Any]: A dictionary containing the updated interactivity state of the generate button.
                """
                return gr.update(
                    interactive=all(
                        [
                            token_value,
                            label_value,
                            category_value,
                            use_case_value,
                            example_value,
                        ]
                    )
                )

            token.change(
                toggle_button,
                inputs=[token, label_boxes, category_boxes, use_case, prompt_examples],
                outputs=generate_button,
            )
            label_boxes.change(
                toggle_button,
                inputs=[token, label_boxes, category_boxes, use_case, prompt_examples],
                outputs=generate_button,
            )
            category_boxes.change(
                toggle_button,
                inputs=[token, label_boxes, category_boxes, use_case, prompt_examples],
                outputs=generate_button,
            )
            use_case.change(
                toggle_button,
                inputs=[token, label_boxes, category_boxes, use_case, prompt_examples],
                outputs=generate_button,
            )
            prompt_examples.change(
                toggle_button,
                inputs=[token, label_boxes, category_boxes, use_case, prompt_examples],
                outputs=generate_button,
            )

            def enable_buttons(state: Dict[str, Any]) -> List[Any]:
                """
                Enables the interactivity of the download and push buttons and loads a preview of the generated data.

                Args:
                    state (Dict[str, Any]): The application state containing the output file path.

                Returns:
                    List[Any]: A list containing:
                               - An update to make the download button interactive.
                               - An update to make the push button interactive.
                               - A DataFrame preview of the generated data.
                """
                df = pd.read_csv(state["output_path"]).head()
                return [gr.update(interactive=True), gr.update(interactive=True), df]

            generate_button.click(
                run_sdg,
                inputs=[
                    sample_size,
                    model,
                    max_new_tokens,
                    save_reasoning,
                    token,
                    state,
                    label_boxes,
                    use_case,
                    prompt_examples,
                    category_boxes,
                ],
                outputs=[status, state],
            ).success(
                enable_buttons,
                inputs=[state],
                outputs=[download_button, push_button, dataframe],
            )

            def download_csv(state: Dict) -> str:
                """
                Generate the file path for downloading a CSV file.

                Args:
                    state (Dict): The application state.

                Returns:
                    str: The file path to the CSV file for download.
                """
                return state[
                    "output_path"
                ]  # Return the file path to trigger the download

            def push_to_huggingface(
                repo_id: str,
                token_value: str,
                is_public: bool,
                state: Dict,
            ) -> str:
                """
                Pushes the generated synthetic data file to the Hugging Face Hub.

                Args:
                    repo_id (str): The ID of the Hugging Face repository (e.g., "username/repo-name").
                    token_value (str): The Hugging Face token for authentication.
                    is_public (bool): Whether to make the repository public.
                    state (Dict): The application state containing the output file path and timestamp.

                Returns:
                    str: A message indicating the result of the upload process.
                """
                try:
                    api = HfApi(token=token_value)
                except Exception as e:
                    return f"Invalid token for writing to Hugging Face: {e}"

                try:
                    # Ensure the repository exists, creating it if it doesn't
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        exist_ok=True,
                        private=not is_public,
                    )

                    api.upload_file(
                        path_or_fileobj=state["output_path"],
                        path_in_repo=f"{state['timestamp']}.csv",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                except Exception as e:
                    return f"Error uploading file to Hugging Face: {e}"
                visibility = "public" if is_public else "private"
                return f"File pushed to {visibility} Hugging Face Hub at {repo_id}/{state['timestamp']}.csv"

            download_button.click(download_csv, inputs=state, outputs=file_output).then(
                lambda: gr.update(visible=True), outputs=file_output
            )

            push_button.click(
                push_to_huggingface,
                inputs=[repo_id, token, is_public_repo, state],
                outputs=status,
            )

        with gr.Tab("About"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(
                        "../../polite-guard.png",
                        show_download_button=False,
                        show_fullscreen_button=False,
                        show_label=False,
                        show_share_button=False,
                        container=False,
                    )
                with gr.Column(scale=3):
                    gr.Markdown(
                        """
                        This synthetic data generator, part of Intel's [Polite Guard](https://huggingface.co/Intel/polite-guard) project, utilizes a specified language model to generate synthetic data for a given use case. 
                        If you find this project valuable, please consider giving it a ❤️ on Hugging Face and sharing it with your network.
                        Visit 
                        - [Polite Guard GitHub repository](https://github.com/intel/polite-guard) for the source code that you can run through the command line on an AI PC or [Intel Tiber AI Cloud](https://ai.cloud.intel.com/), 
                        - [Synthetic Data Generation with Language Models: A Practical Guide](https://medium.com/p/0ff98eb226a1) to learn more about the implementation of this data generator, and
                        - [Polite Guard Dataset](https://huggingface.co/datasets/Intel/polite-guard) for an example of a dataset generated using this data generator.

                        ## Privacy Notice
                        Please note that this data generator uses AI technology and you are interacting with a chat model. 
                        Prompts that are being used during the demo and your personal information will not be stored. 
                        For information regarding the handling of personal data collected refer to the Global Privacy Notice (https://www.intel.com/content/www/us/en/privacy/intelprivacy-notice.html), which encompass our privacy practices.
                        """
                    )

    demo.launch()


if __name__ == "__main__":
    main()
