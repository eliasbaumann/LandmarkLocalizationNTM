import collections
import tensorflow as tf

import adressing
import util

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


def _erase_and_write(memory, address, reset_weights, values):
    with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
        expand_address = tf.expand_dims(address, 3)
        reset_weights = tf.expand_dims(reset_weights, 2)
        weighted_resets = expand_address * reset_weights
        reset_gate = util.reduce_prod(1 - weighted_resets, 1)
        memory *= reset_gate

    with tf.name_scope('additive_write', values=[memory, address, values]):
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix

    return memory


class MemoryAccess(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, name='memory_access'):
        super(MemoryAccess, self).__init__()
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._name = name

        self._write_content_weights_mod = adressing.CosineWeights(num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = adressing.CosineWeights(num_reads, word_size, name='read_content_weights')

        self._linkage = adressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = adressing.Freeness(memory_size)

    def __call__(self, x, prev_state):
        inputs = self._read_inputs(x)
        usage = self._freeness(
            write_weights=prev_state.write_weights,
            free_gate=inputs['free_gate'],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage)
        
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = _erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=inputs['erase_vectors'],
            values=inputs['write_vectors'])

        linkage_state = self._linkage(write_weights, prev_state.linkage)

        # Read from memory.
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weights,
            link=linkage_state.link)
        read_words = tf.matmul(read_weights, memory)

        return (read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage))

    def _read_inputs(self, inputs):
        def _linear(first_dim, second_dim, name, activation=None):
            """Returns a linear transformation of `inputs`, followed by a reshape."""
            linear = tf.keras.layers.Dense(first_dim * second_dim, name=name)(inputs)
            if activation is not None:
                linear = activation(linear, name=name + '_activation')
            return tf.reshape(linear, [-1, first_dim, second_dim])

        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors',
                                tf.sigmoid)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gate = tf.sigmoid(
            tf.keras.layers.Dense(self._num_reads, name='free_gate')(inputs))

        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        allocation_gate = tf.sigmoid(
            tf.keras.layers.Dense(self._num_writes, name='allocation_gate')(inputs))

        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = tf.sigmoid(
            tf.keras.layers.Dense(self._num_writes, name='write_gate')(inputs))

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        num_read_modes = 1 + 2 * self._num_writes
        read_mode = tf.nn.softmax(
            _linear(self._num_reads, num_read_modes, name='read_mode'))

        # Parameters for the (read / write) "weights by content matching" modules.
        write_keys = _linear(self._num_writes, self._word_size, 'write_keys')
        write_strengths = tf.keras.layers.Dense(self._num_writes, name='write_strengths')(
            inputs)

        read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
        read_strengths = tf.keras.layers.Dense(self._num_reads, name='read_strengths')(inputs)

        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_mode': read_mode,
        }
        return result

    def _write_weights(self, inputs, memory, usage):
        with tf.name_scope('write_weights', values=[inputs, memory, usage]):
        # c_t^{w, i} - The content-based weights for each write head.
            write_content_weights = self._write_content_weights_mod(memory, inputs['write_content_keys'], inputs['write_content_strengths'])

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self._freeness.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self._num_writes)

            # Expands gates over memory locations.
            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights + (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        with tf.name_scope(
            'read_weights', values=[inputs, memory, prev_read_weights, link]):
        # c_t^{r, i} - The content weightings for each read head.
            content_weights = self._read_content_weights_mod(
                memory, inputs['read_content_keys'], inputs['read_content_strengths'])

            # Calculates f_t^i and b_t^i.
            forward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=True)
            backward_weights = self._linkage.directional_read_weights(
                link, prev_read_weights, forward=False)

            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = (
                inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

            read_weights = (
                tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(
                    tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
                tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))

            return read_weights

    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=tf.TensorShape([self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            linkage=self._linkage.state_size,
            usage=self._freeness.state_size)

    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._num_reads, self._word_size])