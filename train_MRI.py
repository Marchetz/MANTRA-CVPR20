import argparse
import trainer_mem


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=10)
    parser.add_argument("--model", default='pretrained_models/model_AE2019-05-23 15:50:06')
    parser.add_argument("--memory_past", default='pretrained_models/memory_past.pt')
    parser.add_argument("--memory_fut", default='pretrained_models/memory_fut.pt')
    parser.add_argument("--dim_embedding_key", type=int, default=48)

    parser.add_argument("--rotation_aug", default=True)
    parser.add_argument("--saveImages", default=False)
    parser.add_argument("--saveImages_All", default=False)
    parser.add_argument("--info", type=str, default='')
    return parser.parse_args()


def main(config):
    t = trainer_mem.Trainer(config)
    print('start training')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)

