import argparse
from trainer import trainer_pretrain


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=1000)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)

    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--track_file", default="world_traj_kitti_with_intervals_correct.json")
    return parser.parse_args()


def main(config):
    t = trainer_pretrain.Trainer(config)
    print('start training autoencoder')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
