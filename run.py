import tensorflow as tf
import dnccell
import data

import argparse

parser = argparse.ArgumentParser()

# Task
parser.add_argument('--dataset', type=str, default='cephal', help='TODO')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

# Model parameters
parser.add_argument('--hidden_size', type=int, default=64, help='Size of LSTM hidden layer.')
parser.add_argument('--memory_size', type=int, default=16, help='The number of memory slots.')
parser.add_argument('--word_size', type=int, default=16, help='The width of each memory slot.')
parser.add_argument('--num_write_heads', type=int, default=1, help='Number of memory write heads.')
parser.add_argument('--num_read_heads', type=int, default=4, help='Number of memory read heads.')
parser.add_argument('--clip_value', type=int, default=20,
                        help='Maximum absolute value of controller and dnc outputs.')

# Optimizer parameters.
parser.add_argument('--max_grad_norm', type=float, default=50, help='Gradient clipping norm limit.')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate.')
parser.add_argument('--optimizer_epsilon', type=float, default=1e-10,
                      help='Epsilon used for RMSProp optimizer.')

# Training options.
parser.add_argument('--num_training_iterations', type=int, default=100000,
                        help='Number of iterations to train for.')
parser.add_argument('--report_interval', type=int, default=100,
                        help='Iterations between reports (samples, valid loss).')
parser.add_argument('--checkpoint_dir', type=str, default='/tmp/tf/dnc',
                       help='Checkpointing directory.')
parser.add_argument('--checkpoint_interval', type=int, default=-1,
                        help='Checkpointing step interval.')

args = parser.parse_args()

class dnc_model(tf.keras.Model):
    def __init__(self, output_size):
        super(dnc_model, self).__init__()
        self.output_size = output_size
    
    def call(self, inputs):
        access_config = {
            "memory_size": args.memory_size,
            "word_size": args.word_size,
            "num_reads": args.num_read_heads,
            "num_writes": args.num_write_heads}
        controller_config = {
            "hidden_size": args.hidden_size}
        clip_value = args.clip_value
        cell = dnccell.DNCCell(access_config, controller_config, self.output_size, clip_value)
        initial_state = cell.initial_state(args.batch_size)
        output_sequence, _ = tf.keras.layers.RNN(cell, time_major=True, initial_state=initial_state)(inputs)
        return output_sequence
        
def loss_func(value,label):
    return 0

def train(num_training_iterations, report_interval):
    count = 0
    dataset = data.Data_Loader(args.dataset, args.batch_size)()
    iterator = iter(dataset.data)
    model = dnc_model(dataset.output_size)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.learning_rate,epsilon=args.optimizer_epsilon)
    for iteration in num_training_iterations:
        observation, label = next(iterator)
        with tf.GradientTape() as tape:
            output_logits = model(observation)
            loss = loss_func(output_logits, label)
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_weights), args.max_grad_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        count +=1
        if count % report_interval == 0:
            print(loss)

if __name__ == "__main__":
    train(args.num_training_iterations, args.report_interval)