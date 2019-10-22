import argparse
from trainer import trainer_IRM


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--model", default='pretrained_models/model_decoder_FT/model_FTdecoder2019-09-19 09:48:05')
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--saveImages", default=True)
    parser.add_argument("--saveImages_All", default=False)
    parser.add_argument("--info", type=str, default='')
    return parser.parse_args()

def main(config):
    t = trainer_IRM.Trainer(config)
    print('start training IRM')
    t.fit()

if __name__ == "__main__":
    config = parse_config()
    main(config)

