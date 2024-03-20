from transformers import pipeline
import pandas as pd

classifier = pipeline(task='text-classification')

review = 'what a great movie! i really want to see it multiple times again!'
model_response = classifier(inputs=review)

response = pd.DataFrame(model_response)
print(response)