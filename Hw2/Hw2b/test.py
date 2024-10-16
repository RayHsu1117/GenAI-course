from transformers import pipeline

classifier = pipeline('summerization')
result = classifier(['It is my pleasure to meet you.', 'Due to my homework, I feel pressure.'])
print(result)