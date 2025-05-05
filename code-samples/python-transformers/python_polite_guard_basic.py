from transformers import pipeline

classifier = pipeline("text-classification", "Intel/polite-guard")
text = "You look good today"
output = classifier(text)
print(output)
