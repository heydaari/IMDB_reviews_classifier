import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
import keras_nlp
import tensorflow_datasets as tfds

# Load IMDb movie reviews dataset
imdb_train, imdb_test = tfds.load("imdb_reviews",
                                  split=["train", "test"],
                                  as_supervised=True,
                                  batch_size=1)

print(list(imdb_train)[0][0])
print(list(imdb_train)[0][1])
exit()

# Load a BERT model
classifier = keras_nlp.models.BertClassifier.from_preset("bert_base_en_uncased", num_classes=2)

# Fine-tune on IMDb movie reviews
classifier.fit(imdb_train, )

# Predict on new examples
classifier.predict(["What an amazing movie!", "A total waste of my time."])
