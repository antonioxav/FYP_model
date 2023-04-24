import tensorflow as tf
from model.model import create_model
import numpy as np
import pickle

batch_size = 32
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256


def train(ticker, pillar, pca, experiment_name):
    X_train = np.load(f'data/processed/{pillar}/{ticker}_pca_{pca}_X_train.npy')
    Y_train = np.load(f'data/processed/{pillar}/{ticker}_pca_{pca}_Y_train.npy')
    X_val = np.load(f'data/processed/{pillar}/{ticker}_pca_{pca}_X_val.npy')
    Y_val = np.load(f'data/processed/{pillar}/{ticker}_pca_{pca}_Y_val.npy')

    model = create_model(X_train.shape[1:], 'reg', d_k, d_v, n_heads, ff_dim, schedule_lr=True)
    model.summary()

    callback = tf.keras.callbacks.ModelCheckpoint(f'instances/{experiment_name}.hdf5', 
                                                monitor='val_loss', 
                                                save_best_only=True, verbose=1)


    history = model.fit(X_train, Y_train, 
                        batch_size=batch_size, 
                        epochs=75, 
                        callbacks=[callback],
                        validation_data=(X_val, Y_val)) 

    with open(f'instances/{experiment_name}_history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__=='__main__':
    ticker = 'MSFT'
    pillar = 'macro'
    pca = 'all'
    experiment_name = f'{ticker}_{pillar}_reg10_4_pca_all_huber0.1'

    train(ticker, pillar, pca, experiment_name)

