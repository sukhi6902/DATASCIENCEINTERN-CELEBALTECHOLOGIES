# Importing the libraries
from keras.preprocessing.text import Tokenizer
from keras import models, layers
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import pandas as pd
import pickle

# Importing the pre-processed excel file
dataset = pd.read_csv('cleaned_train.csv')

dataset = dataset.fillna('empty')

# Seperate the feature columns and target column
features = dataset[['cleaned_comment_text']]
target = dataset[['toxic', 'severe_toxic',
                 'obscene', 'threat', 'insult', 'identity_hate']]

# Model Creation - LSTM (Long Short-Term Memory)

# Creating a Tokenizer object with a vocabulary size of 1000 and an out-of-vocabulary (oov) token 'UNK'
tok = Tokenizer(num_words=1000, oov_token='UNK')
# Fitting the Tokenizer on the 'comment_text' column of the 'dataset' DataFrame
#tok.fit_on_texts(dataset['cleaned_comment_text'])
tok.fit_on_texts(dataset['cleaned_comment_text'].values)

# Converting the text data in 'comment_text' column of the 'dataset' DataFrame into sequences of integers
x_train = tok.texts_to_sequences(dataset['cleaned_comment_text'])

# Determining the vocabulary size based on the number of unique words in the Tokenizer's word_index, and adding 1 to account for the 'UNK' token
vocab_size = len(tok.word_index) + 1

# Pad the training sequences to a maximum length of 50
training_padded = pad_sequences(x_train,
                                maxlen=50,
                                truncating='post',
                                padding='post'
                                )

# Create a Sequential model
model = models.Sequential()

# Add an Embedding layer to map tokenized words to dense vectors
# Input length is set to 50 to handle sequences of maximum length 50 (padded sequences)
model.add(layers.Embedding(vocab_size, 128, input_length=50))

# Add the first LSTM layer with 512 units and dropout regularization
# Return_sequences=True is set to pass the output sequence to the next LSTM layer
model.add(layers.LSTM(512, dropout=0.2,
          recurrent_dropout=0.2, return_sequences=True))

# Add the second LSTM layer with 128 units and dropout regularization
model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Add a Dense layer with 16 units and ReLU activation
model.add(layers.Dense(16, activation='relu'))

# Add the output layer with 6 units and Sigmoid activation for multi-label classification
model.add(layers.Dense(6, activation='sigmoid'))

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(
    loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'])

history = model.fit(training_padded,
                    dataset[['toxic', 'severe_toxic', 'obscene',
                             'threat', 'insult', 'identity_hate']],
                    epochs=5,
                    batch_size=512,
                    validation_split=0.2)


# Saving model to disk using pickle
pickle.dump(model, open('model.pkl', 'wb'))


# FOR TESTING

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[12.3,4.5,34.6,56.7,7.6,12.3,12.5]]))
