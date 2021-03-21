from typing import List
import io

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from flax.metrics import tensorboard

MAX_ROWS = 15
MAX_COLUMNS = 15
SHOW_NUMBERS = False

def plot_attention(input_sequence: List[str],
                   output_sequence: List[str],
                   scores: np.array):

  title = 'Attention'
  if len(output_sequence) > MAX_COLUMNS:
    output_sequence = output_sequence[:MAX_COLUMNS]
    scores = scores[:,:MAX_COLUMNS]
    title = 'Attention (truncated columns)'

  if len(input_sequence) > MAX_ROWS:
    input_sequence = input_sequence[:MAX_ROWS]
    scores = scores[:MAX_ROWS,:]
    title = 'Attention (truncated rows)'

  fig, ax = plt.subplots()
  # Create heatmap from scores.
  im = ax.imshow(scores, cmap=plt.cm.Blues, alpha=0.9)

  # Add input and output sequence as labels.
  ax.set_xticks(np.arange(len(output_sequence)))
  ax.set_xticklabels(output_sequence)
  ax.set_yticks(np.arange(len(input_sequence)))
  ax.set_yticklabels(input_sequence)

  # Rotate and shift output sequence labels.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Add the scores for each cell.
  if SHOW_NUMBERS:
    for i in range(len(input_sequence)):
      for j in range(len(output_sequence)):
        score = "%.2f" % scores[i, j]
        text = ax.text(j, i, score, ha="center", va="center", color="y")

  ax.set_title(title)
  # fig.tight_layout()
  plt.show()

  # Convert plot to np array.
  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = canvas.buffer_rgba()
  np_img = np.asarray(buf)

  plt.close(fig)

  return np_img

def save_attention_img_to_tensorboard(summary_writer: tensorboard.SummaryWriter,
                                      step: int,
                                      input_sequence: List[str],
                                      output_sequence: List[str],
                                      scores: np.array):

  attention_image = plot_attention(input_sequence, output_sequence, scores)
  summary_writer.image('Attention', attention_image, step)
