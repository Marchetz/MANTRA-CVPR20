import argparse
from trainer import trainer_pretrain_map

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    return parser.parse_args()

def main(config):
    t = trainer_pretrain_map.Trainer(config)
    print('start training autoencoder')
    t.fit()

if __name__ == "__main__":
    config = parse_config()
    main(config)
