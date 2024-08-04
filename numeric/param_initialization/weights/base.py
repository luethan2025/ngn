#!/usr/bin/env python

class WeightInitializer:
  """Base class for weight initialization."""
  def init_params(self):
    """Initialize weight matrix.

    Returns
    -------
    np.array
      Initialized weight matrix.
    """
    raise NotImplementedError()
