import argparse
from trainer import trainer_controllerMem


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--model_ae", default='pretrained_models/model_AE/model_ae')
    parser.add_argument("--track_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will use in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    print('Start training writing controller')
    t = trainer_controllerMem.Trainer(config)
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
