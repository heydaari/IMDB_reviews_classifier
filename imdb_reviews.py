# Developed by Mohammad Hassan Heydari
# IMDB Reviews classifier
import keras.layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, Flatten
from numpy import array

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']
train_sentences, train_labels = [], []
test_sentences, test_labels = [], []

for s, l in train_data :
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

train_labels = array(train_labels)
test_labels = array(test_labels)


tokenizer = Tokenizer(num_words= 10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=120, truncating='post')
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=120, truncating='post')

model = Sequential([
    Embedding(10000, 16, input_length=120),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')

])

model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

