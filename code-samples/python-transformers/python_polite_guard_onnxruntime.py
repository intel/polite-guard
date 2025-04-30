from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("intel/polite-guard")
model = ORTModelForSequenceClassification.from_pretrained("intel/polite-guard", file_name="onnx/model_quantized.onnx")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

output = classifier("You look good today")
print(output)
