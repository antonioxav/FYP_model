import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from model.backbone import get_backbone
from model.positional_encoding import Time2Vector
from model.transformer import TransformerEncoder

def create_model(shape, task, d_k, d_v, n_heads, ff_dim, schedule_lr = False):
  '''Initialize time and transformer layers'''
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  backbone = get_backbone('linear', 4)

  '''Construct model'''
  in_seq = Input(shape=shape)
  in_seq_rep = backbone(in_seq)

  time_embedding = Time2Vector(in_seq_rep.shape[-2])
  pos = time_embedding(in_seq_rep)
  x = Concatenate(axis=-1)([in_seq_rep, pos])

  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
  x = Dropout(0.3)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.3)(x)
  out = Dense(1, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  loss = {
    'bc': tf.keras.losses.BinaryCrossentropy(from_logits=True),
    'reg': tf.keras.losses.Huber(delta=0.1) 
    # 'reg': tf.keras.losses.MeanSquaredError()
  }

  lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 256*10, 0.95, staircase=False, name=None)
  adam = tf.keras.optimizers.Adam(learning_rate=lr_scheduler if schedule_lr else 0.001)

  metrics = {
    'bc': [tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    'reg': ['mae', 'mape']
  }

  model.compile(loss=loss[task], optimizer=adam, metrics=metrics[task])
  return model

