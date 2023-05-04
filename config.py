import argparse


# define global parameters
def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication system")

    # parameter of datasets
    parser.add_argument("--image_height", type=int, default=28, help="height of training and validation images")
    parser.add_argument("--image_width", type=int, default=28, help="width of training and validation images")
    parser.add_argument("--image_channel", type=int, default=1, help="channel of training and validation images")
    parser.add_argument("--val_rate", type=float, default=0.5, help="sample rate for validation")
    # parameter of model
    parser.add_argument("--num_symbol_node", type=int, default=256, help="the number of symbols sent by the node.")
    parser.add_argument("--num_symbol_relay", type=int, default=256, help="the number of symbols sent by relay.")
    # parameter of wireless channel
    parser.add_argument("--channel_type", type=str, default='AWGN', help="channel type during training。Rayleigh")
    parser.add_argument("--snr_train_dB_up", type=int, default=7, help="snr of node to relay in dB for training.")
    parser.add_argument("--snr_train_dB_down", type=int, default=7, help="snr of relay to node in dB for training.")
    # parameter of training
    parser.add_argument("--train_channel", type=str, default='gen_channel',
                        help="training transmitter and receiver with generate channel or real channel.")
    parser.add_argument("--num_epochs", type=int, default=100, help="training epochs.")
    parser.add_argument("--number_steps_channel", type=int, default=1, help="the number of training channel.")
    parser.add_argument("--number_steps_transmitter", type=int, default=1, help="the number of training transmitter.")
    parser.add_argument("--number_steps_receiver", type=int, default=1, help="the number of training receiver.")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate.")
    parser.add_argument("--lr_decay", type=bool, default=False, help="whether to use decreasing learning rate.")
    parser.add_argument("--noise_dim", type=int, default=2, help="the dimension of noise.")

    args = parser.parse_args()

    return args