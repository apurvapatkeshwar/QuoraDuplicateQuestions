from __future__ import print_function
import numpy as np
import sys, csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import * #Dot, Input, Bidirectional, GRU,LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization, concatenate, Lambda, Permute, Concatenate, Multiply
from keras.activations import softmax
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation
import nltk
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

nltk.download('stopwords')

def unchanged_shape(input_shape):
    return input_shape

def substract(input_1, input_2):
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_

def submult(input_1, input_2):
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_

def apply_multiple(input_, layers):
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    #text = unicode(text, "utf-8")
    text = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return(text)

# Initialize global variables
KERAS_DATASETS_DIR = '' 
QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
GLOVE_FILE = 'glove.840B.300d.txt'
Q1_TRAINING_DATA_FILE = 'q1_train_bilstm.npy'
Q2_TRAINING_DATA_FILE = 'q2_train_bilstm.npy'
LABEL_TRAINING_DATA_FILE = 'label_train_bilstm.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix_bilstm.npy'
NB_WORDS_DATA_FILE = 'nb_words_bilstm.json'
MAX_SEQUENCE_LENGTH = 30
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5.gru.withattention'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NB_EPOCHS = 25 
DROPOUT = 0.1 #Hyperparameter
BATCH_SIZE = 500 #Hyperparameter
WORD_EMBEDDING_DIM = 300
SENT_EMBEDDING_DIM = 200
RNG_SEED = 13371447
OPTIMIZER = 'adam'
MAX_NB_WORDS = 2000000

# If the dataset, embedding matrix and word count exist in the local directory
if exists(Q1_TRAINING_DATA_FILE) and exists(Q2_TRAINING_DATA_FILE) and exists(LABEL_TRAINING_DATA_FILE) and exists(NB_WORDS_DATA_FILE) and exists(WORD_EMBEDDING_MATRIX_FILE):
    # Then load them
    q1_data = np.load(open(Q1_TRAINING_DATA_FILE, 'rb'))
    q2_data = np.load(open(Q2_TRAINING_DATA_FILE, 'rb'))
    labels = np.load(open(LABEL_TRAINING_DATA_FILE, 'rb'))
    word_embedding_matrix = np.load(open(WORD_EMBEDDING_MATRIX_FILE, 'rb'))
    with open(NB_WORDS_DATA_FILE, 'r') as f:
        nb_words = json.load(f)['nb_words']
else:
    # Else download and extract questions pairs data
    if not exists(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE):
        get_file(QUESTION_PAIRS_FILE, QUESTION_PAIRS_FILE_URL)

    print("Processing", QUESTION_PAIRS_FILE)

    question1 = []
    question2 = []

    is_duplicate = []
    c = 0
    with open(KERAS_DATASETS_DIR + QUESTION_PAIRS_FILE) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
              #question1.append(text_to_wordlist(row['question1'], remove_stopwords=True, stem_words=True).encode('ascii', 'ignore'))
              #question2.append(text_to_wordlist(row['question2'], remove_stopwords=True, stem_words=True).encode('ascii', 'ignore'))
              question1.append(row['question1'])
              question2.append(row['question2'])
              is_duplicate.append(row['is_duplicate'])

    print('Question pairs: %d' % len(question1))

    # Build tokenized word index
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    word_index = tokenizer.word_index

    print("Words in index: %d" % len(word_index))

    # Download and process GloVe embeddings
    if not exists(KERAS_DATASETS_DIR + GLOVE_ZIP_FILE):
        zipfile = ZipFile(get_file(GLOVE_ZIP_FILE, GLOVE_ZIP_FILE_URL))
        zipfile.extract(GLOVE_FILE, path=KERAS_DATASETS_DIR)

    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(KERAS_DATASETS_DIR + GLOVE_FILE) as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))

    # Prepare word embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
        else:
            word_embedding_matrix[i] = np.random.rand(1,300)
        
    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

    # Prepare training data tensors
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)

    # Persist training and configuration data to files
    np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), q1_data)
    np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), q2_data)
    np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)

sys.stdout.flush()
# Partition the dataset into train and test sets
X = np.stack((q1_data, q2_data), axis=1)
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]

print("Q1_train.shape: ", Q1_train.shape)
print("Q2_train.shape :", Q2_train.shape)
print("Q1_test.shape: ", Q1_test.shape)
print("Q2_test.shape :", Q2_test.shape)

# Define the model
question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
q1 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)
print("q1 shape :", q1.shape)
q1 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=True))(q1)

print("q1 shape :", q1.shape)
q2 = Embedding(nb_words + 1, 
                 WORD_EMBEDDING_DIM, 
                 weights=[word_embedding_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)
print("q2 shape :", q2.shape)
q2 = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=True))(q2)
attention = Dot(axes=-1)([q1, q2])
w_att_1 = Lambda(lambda x: softmax(x, axis=1), output_shape=unchanged_shape)(attention)
w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(attention))
q1_aligned = Dot(axes=1)([w_att_1, q1])
q2_aligned = Dot(axes=1)([w_att_2, q2])

q1_combined = concatenate([q1, q2_aligned, submult(q1, q2_aligned)])
q2_combined = concatenate([q2, q1_aligned, submult(q2, q1_aligned)]) 
compose = Bidirectional(GRU(SENT_EMBEDDING_DIM, return_sequences=True))
q1_compare = compose(q1_combined)
q2_compare = compose(q2_combined)

q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])


merged = concatenate([q1_rep, q2_rep])
merged = BatchNormalization()(merged)
merged = Dense(1000, activation='elu')(merged)

merged = BatchNormalization()(merged)
merged = Dense(500, activation='elu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='elu')(merged)
merged = BatchNormalization()(merged)
merged = Dense(100, activation='elu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
#print("Starting training at", datetime.datetime.now())
#sys.stdout.flush()
#t0 = time.time()
#callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
#history = model.fit([Q1_train, Q2_train],
#                    y_train,
#                    epochs=NB_EPOCHS,
#                    validation_split=VALIDATION_SPLIT,
#                    verbose=2,
#                    batch_size=BATCH_SIZE,
#                    callbacks=callbacks)
#t1 = time.time()
#print("Training ended at", datetime.datetime.now())
#print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

# Print best validation accuracy and epoch
#max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
#print('Maximum validation accuracy = {0:.4f} (epoch {1:d})'.format(max_val_acc, idx+1))

# Evaluate the model with best validation accuracy on the test partition
model.load_weights(MODEL_WEIGHTS_FILE)
loss, accuracy, f1 = model.evaluate([Q1_test, Q2_test], y_test, verbose=1)
print(model.evaluate([Q1_test, Q2_test], y_test, verbose=1))
print('Test loss = {0:.4f}, test accuracy = {1:.4f}, F1 score = {1:.4f}'.format(loss, accuracy, f1))
sys.stdout.flush()
