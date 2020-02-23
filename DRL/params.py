import argparse

parser = argparse.ArgumentParser(description='DD_DRL')
# -- Basic --
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--eps', type=float, default=1e-6,
                    help='epsilon (default: 1e-6)')
parser.add_argument('--num_proc', type=int, default=1,
                    help='number of processors (default: 1)')
parser.add_argument('--num_exp', type=int, default=10,
                    help='number of experiments (default: 10)')

# learning
parser.add_argument('--node_input_dim', type=int, default=5,
                    help='node input dimensions to graph embedding (default: 5)')
parser.add_argument('--job_input_dim', type=int, default=3,
                    help='job input dimensions to graph embedding (default: 3)')
parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=8,
                    help='Maximum depth of root-leaf message passing (default: 8)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--ba_size', type=int, default=64,
                    help='Batch size (default: 64)')
parser.add_argument('--gamma', type=float, default=1,
                    help='discount factor (default: 1)')
parser.add_argument('--early_terminate', type=int, default=0,
                    help='Terminate the episode when stream is empty (default: 0)')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--log_file_name', type=str, default='log',
                    help='log file name (default: log)')
parser.add_argument('--master_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in master (default: 0)')
parser.add_argument('--worker_num_gpu', type=int, default=0,
                    help='Number of GPU cores used in worker (default: 0)')
parser.add_argument('--master_gpu_fraction', type=float, default=0.5,
                    help='Fraction of memory master uses in GPU (default: 0.5)')
parser.add_argument('--worker_gpu_fraction', type=float, default=0.1,
                    help='Fraction of memory worker uses in GPU (default: 0.1)')
parser.add_argument('--average_reward_storage_size', type=int, default=100000,
                    help='Storage size for computing average reward (default: 100000)')
parser.add_argument('--reset_prob', type=float, default=0,
                    help='Probability for episode to reset (after x seconds) (default: 0)')
parser.add_argument('--reset_prob_decay', type=float, default=0,
                    help='Decay rate of reset probability (default: 0)')
parser.add_argument('--reset_prob_min', type=float, default=0,
                    help='Minimum of decay probability (default: 0)')
parser.add_argument('--num_agents', type=int, default=1,
                    help='Number of parallel agents (default: 16)')
parser.add_argument('--num_ep', type=int, default=10000000,
                    help='Number of training epochs (default: 10000000)')
parser.add_argument('--learn_obj', type=str, default='mean',
                    help='Learning objective (default: mean)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--check_interval', type=float, default=0.01,
                    help='interval for master to check gradient report (default: 10ms)')
parser.add_argument('--model_save_interval', type=int, default=1000,
                    help='Interval for saving Tensorflow model (default: 1000)')
parser.add_argument('--num_saved_models', type=int, default=1000,
                    help='Number of models to keep (default: 1000)')

args = parser.parse_args()
