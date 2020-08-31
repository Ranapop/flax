# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Module with main function"""

import os
from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from jax.config import config

import input_pipeline as inp
import train

# hyperparam defaults
DEFAULT_BATCH_SIZE = 2048
NUM_EPOCHS = 750

config.enable_omnistaging()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # or pass as env var
tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate',
                   default=0.003,
                   help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer('batch_size',
                     default=DEFAULT_BATCH_SIZE,
                     help=('Batch size for training.'))

flags.DEFINE_integer('num_epochs',
                     default=NUM_EPOCHS,
                     help=('Number of epochs.'))

flags.DEFINE_integer(
    'max_query_length',
    default=100,  #92 max in the train dataset
    help=('Length of the predicted query.'))

flags.DEFINE_integer('seed',
                     default=0,
                     help=('Random seed for network initialization.'))

flags.DEFINE_integer('model_dir',
                     default=None,
                     help=('Model dir to save model to/load model from.'))                     


def main(_):
    """Load the cfq data and train the model"""
    # prepare data source
    data_source = inp.CFQDataSource(seed=FLAGS.seed,
                                    fixed_output_len=False,
                                    cfq_split='random_split')

    # train model
    trained_model = train.train_model(
        learning_rate=FLAGS.learning_rate,
        num_epochs=2,
        max_out_len=FLAGS.max_query_length,
        # num_epochs=FLAGS.num_epochs,
        seed=FLAGS.seed,
        data_source=data_source,
        batch_size=FLAGS.batch_size,
        bucketing=True)


if __name__ == '__main__':
    app.run(main)
