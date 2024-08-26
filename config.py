import argparse


def get_configuration():
    parser = argparse.ArgumentParser(description='Using the pypots library for time series imputation.')

    # 训练参数
    parser.add_argument('--seq-length', dest='seq_length', type=int, default=64, help='Specified sequence length.')
    parser.add_argument('--missing-rate', dest='missing_rate', type=float, default=0.8, help='Set missing_rate.')
    parser.add_argument('--max-missing-rate', dest='max_missing_rate', type=float, default=0.25,
                        help='Set max_missing_rate.')

    args = parser.parse_args()
    args.batch_size = 64
    args.epochs = 50

    return args



if __name__ == '__main__':
    pass
