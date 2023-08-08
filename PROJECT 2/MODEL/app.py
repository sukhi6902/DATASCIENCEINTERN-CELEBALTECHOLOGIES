import numpy as np
from flask import Flask, request, render_template
import pickle
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

# initiating the flask app
app = Flask(__name__)
# reading the pickled regression model
model = pickle.load(open('model.pkl', 'rb'))


# redirecting to the html homepage
@app.route('/')
def home():
    return render_template('index.html')


# POST method for sending inofrmation to the site after processing
@app.route('/predict', methods=['POST'])
def predict():
    
    features = [x for x in request.form.values()]
    
    # tokenize
    tok = Tokenizer(num_words=1000, oov_token='UNK')
    # Fitting the Tokenizer on the 'comment_text' column of the 'dataset' DataFrame
    #tok.fit_on_texts(dataset['cleaned_comment_text'])
    tok.fit_on_texts(features)

    # Converting the text data in 'comment_text' column of the 'dataset' DataFrame into sequences of integers
    x_train = tok.texts_to_sequences(features)

    # Determining the vocabulary size based on the number of unique words in the Tokenizer's word_index, and adding 1 to account for the 'UNK' token
    #vocab_size = len(tok.word_index) + 1

    # Pad the training sequences to a maximum length of 50
    training_padded = pad_sequences(x_train,
                                maxlen=50,
                                truncating='post',
                                padding='post'
                                )
    
    training_padded = [np.array(training_padded)]
    
    #final_features = [np.array(float_features)]
    # using model's predict method on the created array from the input data
    prediction = model.predict(training_padded)


    # categorizing the classification in to 1 and 0

    return render_template('index.html', classify_text='The enetred text is classfied into [toxic, severe_toxic, obscene, threat, insult, identity_hate] : {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
