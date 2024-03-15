import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    name="imdb_reviews", split=["train", "test"], as_supervised=True, batch_size=16
)

print(imdb_train[1])
exit()

classifier = keras_nlp.models.BertClassifier.from_preset(
    preset= "bert_base_en_uncased", num_classes=2
)

classifier.fit(imdb_train, validation_data=imdb_test)

# Predict sentiment for new examples
classifier.predict(["What an amazing movie!", "A total waste of my time."])
