import argparse

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Using Wide and Deep Neural Network')

    parser.add_argument('filename', type=str, help="name of file to process")
    # parser.add_argument('-mv', type=str, choices=['wdnn', 'wdnna'])
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'])
        
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--vec_length', type=int, default=150, help='vector dimension')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')

    return parser.parse_args()

