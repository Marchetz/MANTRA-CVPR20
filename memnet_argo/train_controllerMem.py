import argparse
from trainer import trainer_controllerMem_map
import pdb

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=50)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--preds", type=int, default=6)
    parser.add_argument("--th_memory", type=int, default=0.5)
    parser.add_argument("--model", default='pretrained_models/ae/model_epoch_99_2020-01-15 23:11:16')

    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    return parser.parse_args()

def main(config):
    print('Start training writing controller')
    t = trainer_controllerMem_map.Trainer(config)
    t.fit()

if __name__ == "__main__":
    config = parse_config()
    main(config)
