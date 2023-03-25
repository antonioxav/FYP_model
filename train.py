import tensorflow as tf
from model.model import create_model
import numpy as np
import pickle

batch_size = 32
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256


ticker = 'SPY'

X_train = np.load('data/processed/macro/X_train.npy')
Y_train = np.load('data/processed/macro/Y_train.npy')
X_val = np.load('data/processed/macro/X_val.npy')
Y_val = np.load('data/processed/macro/Y_val.npy')

model = create_model(X_train.shape[1:], 'reg', d_k, d_v, n_heads, ff_dim, schedule_lr=True)
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint(f'instances/{ticker}_p3_macro_reg40_all.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, verbose=1)


history = model.fit(X_train, Y_train, 
                    batch_size=batch_size, 
                    epochs=50, 
                    callbacks=[callback],
                    validation_data=(X_val, Y_val)) 

with open(f'instances/{ticker}_p3_macro_reg40_all_history.pickle', 'xb') as file_pi:
    pickle.dump(history.history, file_pi)