# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Implementation of Neural Net (NN) functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import code
import math

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_logging_ops


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.pack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])

def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            adapted_bias=None,
                            context_embeddings=None,
                            name=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.

  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).

  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
  Returns:
    out_logits, out_labels: `Tensor` objects each with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
  """

  if isinstance(weights, variables.PartitionedVariable):
    weights = list(weights)
  if not isinstance(weights, list):
    weights = [weights]

  scope = weights + [biases, inputs, labels]
  if adapted_bias is not None:
    scope += [adapted_bias, context_embeddings]
  with ops.name_scope(name, "compute_sampled_logits", scope):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = sampled_values
    # pylint: enable=unpacking-non-sequence

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat(0, [labels_flat, sampled])

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    all_b = embedding_ops.embedding_lookup(biases, all_ids)

    true_adapt = None
    sampled_adapt = None
    if context_embeddings is not None:
      adapted_bias_t = array_ops.transpose(adapted_bias)
      # context_embeddings is [batch_size x context_embed_size]
      true_adapt_embed = embedding_ops.embedding_lookup(
        adapted_bias_t, labels_flat)
      sampled_adapt_embed = embedding_ops.embedding_lookup(
        adapted_bias_t, sampled)
      # prod is [batch_size * num_ids]
      true_adapt = tf.matmul(context_embeddings, true_adapt_embed, 
                             transpose_b=True)
      sampled_adapt = tf.matmul(context_embeddings, sampled_adapt_embed,
                                transpose_b=True)
      vals = [tf.reduce_min(true_adapt), tf.reduce_max(true_adapt),
              tf.reduce_min(sampled_adapt), tf.reduce_max(sampled_adapt)]
      sampled_adapt = tf.Print(sampled_adapt, vals)

    # true_w shape is [batch_size * num_true, dim]
    # true_b is a [batch_size * num_true] tensor
    true_w = array_ops.slice(
        all_w, [0, 0], array_ops.pack([array_ops.shape(labels_flat)[0], -1]))
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat(0, [[-1, num_true], dim])
    row_wise_dots = math_ops.mul(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = tf.reshape(row_wise_dots,
                                array_ops.concat(0, [[-1], dim]))
    true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b
    if true_adapt is not None:
      true_logits += tf.expand_dims(tf.reduce_sum(true_adapt, 1), 1)

    # Lookup weights and biases for sampled labels.
    #   sampled_w shape is [num_sampled, dim]
    #   sampled_b is a [num_sampled] float tensor
    sampled_w = array_ops.slice(
        all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # sampled_b has shape [num_sampled]
    # Apply X*W'+B, which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(
        inputs, sampled_w, transpose_b=True) + sampled_b
    if sampled_adapt is not None:
      sampled_logits += sampled_adapt

    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(
          math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat(1, [acc_indices_2d, acc_ids_2d_int32],
                                        "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = array_ops.concat(0,
          [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)])
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += sparse_ops.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat(1, [true_logits, sampled_logits])
    # true_logits is a float tensor, ones_like(true_logits) is a float tensor
    # of ones. We then divide by num_true to ensure the per-example labels sum
    # to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat(1, [
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ])

  return out_logits, out_labels


def sampled_softmax_loss(weights,
                         biases,
                         labels,
                         inputs,
                         num_sampled,
                         num_classes,
                         num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         context_embeddings=None,
                         adapted_bias=None,
                         name="sampled_softmax_loss"):
  """Computes and returns the sampled softmax training loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  At inference time, you can compute full softmax probabilities with the
  expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

  See our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)

  Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
  ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  """
  logits, labels = _compute_sampled_logits(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      context_embeddings=context_embeddings,
      adapted_bias=adapted_bias,
      name=name)
  sampled_losses = nn_ops.softmax_cross_entropy_with_logits(labels=labels,
                                                            logits=logits)
  # sampled_losses is a [batch_size] tensor.
  return sampled_losses
