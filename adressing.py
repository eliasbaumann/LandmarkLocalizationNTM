import tensorflow as tf
import collections
import util

_EPSILON = 1e-6

TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
                                              ('link', 'precedence_weights'))


@tf.function
def vector_norms(m):
    return tf.sqrt(tf.reduce_sum(m*m, axis=2, keepdims=True)+_EPSILON)


@tf.function
def weighted_softmax(activations, strengths, strengths_op):
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)
    sharp_activations = activations * transformed_strengths
    return tf.nn.softmax(sharp_activations)


class CosineWeights(tf.keras.layers.Layer):
    def __init__(self,
               num_heads,
               word_size,
               strength_op=tf.nn.softplus,
               name='cosine_weights'):
        super(CosineWeights, self).__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op

    def __call__(self, memory, keys, strengths):
        dot = tf.matmul(keys, memory, adjoint_b=True)
        memory_norms = vector_norms(memory)
        key_norms = vector_norms(keys)
        norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
        similarity = dot / (norm + _EPSILON)
        return weighted_softmax(similarity, strengths, self._strength_op)


class TemporalLinkage(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        super(TemporalLinkage, self).__init__()
        self._memory_size = memory_size
        self._num_writes = num_writes
        self._name = name

    def __call__(self, write_weights, prev_state):
        link = self._link(prev_state.link, prev_state.precedence_weights, prev_state.write_weights)
        precedence_weights = self._precedence_weights(prev_state.prev_precedence_weights, prev_state.write_weights)
        return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        with tf.name_scope('directional_read_weights'):
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes, 1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            return tf.transpose(result, perm=[0, 2, 1, 3])

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        with tf.name_scope('link'):
            batch_size = tf.shape(prev_link)[0]
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link
            return tf.linalg.set_diag(link, tf.zeros(
                [batch_size, self._num_writes, self._memory_size],
                dtype=link.dtype))

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keepdims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights

    @property
    def state_size(self):
        return TemporalLinkageState(link=tf.TensorShape([self._num_writes, self._memory_size, self._memory_size]), precedence_weights=tf.TensorShape([self._num_writes, self._memory_size]),)


class Freeness(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_size, name='freeness'):
        super(Freeness, self).__init__()
        self._memory_size = memory_size
        self._name = name

    def __call__(self, write_weights, free_gate, read_weights, prev_usage):
        write_weights = tf.stop_gradient(write_weights)
        usage = self._usage_after_write(prev_usage, write_weights)
        usage = self._usage_after_read(usage, free_gate, read_weights)
        return usage

    def write_allocation_weights(self, usage, write_gates, num_writes):
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)

    def _usage_after_write(self, prev_usage, write_weights):
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            write_weights = 1 - util.reduce_prod(1 - write_weights, 1)
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        with tf.name_scope('usage_after_read'):
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            phi = util.reduce_prod(1 - free_read_weights, 1, name='phi')
            return prev_usage * phi

    def _allocation(self, usage):

        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = _EPSILON + (1 - _EPSILON) * usage

            nonusage = 1 - usage
            sorted_nonusage, indices = tf.nn.top_k(
                nonusage, k=self._memory_size, name='sort')
            sorted_usage = 1 - sorted_nonusage
            prod_sorted_usage = tf.math.cumprod(sorted_usage, axis=1, exclusive=True)
            sorted_allocation = sorted_nonusage * prod_sorted_usage
            inverse_indices = util.batch_invert_permutation(indices)

            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            return util.batch_gather(sorted_allocation, inverse_indices)

    @property
    def state_size(self):
        """Returns the shape of the state tensor."""
        return tf.TensorShape([self._memory_size])