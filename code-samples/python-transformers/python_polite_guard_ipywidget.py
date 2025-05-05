import ipywidgets as widgets
from IPython.display import display, clear_output
from transformers import pipeline

# Create a Text Box
text_box = widgets.Text(
    placeholder="Enter text here",
    description="Input:"
)

# Create a Button
button = widgets.Button(
    description="Analyze",
    button_style="primary"
)

# Create an Output Widget
output = widgets.Output()

# Function to Run When Button is Clicked
def on_button_click(b):
    with output:
        clear_output(wait=True)  # Clear previous output
        classifier = pipeline("text-classification", "Intel/polite-guard")
        result = classifier(text_box.value)
        print(result)
        #print(f"{output}")
        print(f"\n'{text_box.value}' is {result[0]['label']}")
        #print("Hi")

# Link Button Click to Function
button.on_click(on_button_click)

# Display Widgets
display(text_box, button, output)
