
import tensorflow.compat.v2 as tf

# Do I need to put that copyright thingy here??

AUTOTUNE = tf.data.experimental.AUTOTUNE

BOS = b'<bos>'
EOS = b'<eos>'
UNK = b'<unk>'
PAD = b'<pad>'

QUESTION_KEY = 'question'
QUERY_KEY = 'query'
QUESTION_LEN_KEY = 'question_len'
QUERY_LEN_KEY = 'query_len'

ACC_KEY = 'accuracy'
LOSS_KEY = 'loss'
# hyperparams
NUM_EPOCHS = 750
LSTM_HIDDEN_SIZE = 512
DEFAULT_BATCH_SIZE = 2048