import numpy as np
import pandas as pd
import random
import string
import sys
import io
import os

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


#################

N_GPU = 4
LAYER_COUNT = 3
DROPOUT = 0.2
HIDDEN_LAYERS_DIM = 500
BATCH_SIZE = 80
MAXLEN = 60
EPOCHS = 1


#Harry Potter Txt Files from: http://www.glozman.com/textpages.html
work_dir = 'data/'
text = ''
for index in range(1, 5):
    name = "harry_potter_{index}.txt".format(index=index)
    path = os.path.join(work_dir, name)
    with io.open(path, mode="r", encoding="latin-1") as fd:
        content = fd.read()
        text = ' '.join([text,content])

chars = list(string.printable)
chars.remove('\x0b')
chars.remove('\x0c')
chars_to_ix = {c:i for i,c in enumerate(chars)}

text_size, vocab_size = len(text), len(chars)
print('There are %d total characters in your data.' % (text_size))


text = text.replace('\x96','-')
text = text.replace('\x93','"')
text = text.replace('\x91','\'')
text = text.replace('\x92','\'')
text = text.replace('\x90','')
text = text.replace('\x95','')
text = text.replace('\xad','-')  
text = text.replace('«','')
text = text.replace('»','')
text = text.replace('¦','')
text = text.replace('ü','')
text = text.replace('\\','')


train = text[:int(text_size * .9)]
test = text[int(text_size * .9):]
train_size = len(train)
test_size = len(test)


def batch_generator(text, count):
    """Generate batches for training"""
    while True: # keras wants that for reasons
        for batch_ix in range(count):
            X = np.zeros((BATCH_SIZE, MAXLEN, vocab_size))
            y = np.zeros((BATCH_SIZE, vocab_size))

            batch_offset = BATCH_SIZE * batch_ix

            for sample_ix in range(BATCH_SIZE):
                sample_start = batch_offset + sample_ix
                for s in range(MAXLEN):
                    X[sample_ix, s, chars_to_ix[text[sample_start+s]]] = 1
                y[sample_ix, chars_to_ix[text[sample_start+s+1]]]=1

            yield X, y


def build_model(gpu_count=N_GPU):
    model = Sequential()

    for i in range(LAYER_COUNT):
        model.add(
            LSTM(
                HIDDEN_LAYERS_DIM, 
                return_sequences=True if (i!=(LAYER_COUNT-1)) else False,
                input_shape=(MAXLEN, vocab_size),
            )
        )
        model.add(Dropout(DROPOUT))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model



training_model = build_model(gpu_count=N_GPU)

# Load Weights
training_model.load_weights(
    "4-gpu_BS-100_3-500_dp0.20_60S_epoch01-loss2.9326-val-loss2.4937_weights"
)

train_batch_count = (train_size - MAXLEN) // BATCH_SIZE
val_batch_count = (test_size - MAXLEN) // BATCH_SIZE

# checkpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping
filepath = "./%d-gpu_BS-%d_%d-%s_dp%.2f_%dS_epoch{epoch:02d}-loss{loss:.4f}-val-loss{val_loss:.4f}_weights" % (
    N_GPU,
    BATCH_SIZE,
    LAYER_COUNT,
    HIDDEN_LAYERS_DIM,
    DROPOUT,
    MAXLEN
)
checkpoint = ModelCheckpoint(
    filepath,
    save_weights_only=True
)
# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

callbacks_list = [checkpoint, early_stopping]



history = training_model.fit_generator(
    batch_generator(train, count=train_batch_count),
    train_batch_count,
    max_queue_size=1, # no more than one queued batch in RAM
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=batch_generator(test, count=val_batch_count),
    validation_steps=val_batch_count,
    initial_epoch=0
)          






