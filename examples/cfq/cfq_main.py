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
import train_syntax_based

# stage out as much as possible to XLA, not only computations depending
# on arguments, see https://github.com/google/jax/pull/3370
config.enable_omnistaging()
# "magic commands" to make sure jax doesn't take too much memory
# that cuBLAS can't load its kernels into memory.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # or pass as env var
# prevent tf from using the GPU
tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate',
                   default=0.003,
                   help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer('batch_size',
                     short_name='b',
                     default=2048,
                     help=('Batch size for training.'))

flags.DEFINE_integer('num_train_steps',
                     short_name='s',
                     default=35000,
                     help=('Number of train steps.'))

flags.DEFINE_integer('eval_frequency',
                     short_name='e',
                     default=100,
                     help=('At how many steps evaluation is performed.'))

flags.DEFINE_integer(
    'max_query_length',
    default=100,  #92 max in the train dataset
    help=('Length of the predicted query.'))

flags.DEFINE_integer('seed',
                     default=0,
                     help=('Random seed for network initialization.'))

flags.DEFINE_boolean('only_run_test',
                     short_name='t',
                     default=False,
                     help=('Boolean flag indicating wheter to test model.'))

flags.DEFINE_boolean('use_bucketing',
                     default=True,
                     help=('Use bucketing when batching'))

flags.DEFINE_boolean('dummy_data',
                     short_name='d',
                     default=False,
                     help=('Use dummy dataset (reversed sequences)'))

flags.DEFINE_string('model_dir',
                    default='temp/current_model',
                    help=('Model dir to save model to/load model from.'))

flags.DEFINE_string('cfq_split',
                    default='random_split',
                    help=('Cfq split (random_split, mcd1 etc.).'))

flags.DEFINE_boolean(
  'syntax_based',
  default=False,
  help=('Train the syntax based model instead of the baseline.'))

def main(_):
  """Load the cfq data and train the model"""

  if FLAGS.syntax_based:
    train_fn = train_syntax_based.train_model
    test_fn = train_syntax_based.test_model
    #TODO: replace with Seq2Tree once model ready.
    data_source = inp.Seq2SeqCfqDataSource(
      seed=FLAGS.seed,
      fixed_output_len=False,
      cfq_split=FLAGS.cfq_split,
      replace_with_dummy=FLAGS.dummy_data)
  else:
    train_fn = train.train_model
    test_fn = train.test_model
    data_source = inp.Seq2SeqCfqDataSource(
      seed=FLAGS.seed,
      fixed_output_len=False,
      cfq_split=FLAGS.cfq_split,
      replace_with_dummy=FLAGS.dummy_data)

  if FLAGS.only_run_test:
    test_fn(model_dir=FLAGS.model_dir,
            data_source=data_source,
            max_out_len=FLAGS.max_query_length,
            seed=FLAGS.seed,
            batch_size=FLAGS.batch_size)
  else:
    # train model
    trained_model = train_fn(learning_rate=FLAGS.learning_rate,
                             num_train_steps=FLAGS.num_train_steps,
                             max_out_len=FLAGS.max_query_length,
                             seed=FLAGS.seed,
                             data_source=data_source,
                             batch_size=FLAGS.batch_size,
                             bucketing=FLAGS.use_bucketing,
                             model_dir=FLAGS.model_dir,
                             eval_freq=FLAGS.eval_frequency)


if __name__ == '__main__':
  app.run(main)
