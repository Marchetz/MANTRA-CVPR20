import argparse
import evaluate_model
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=50)
    parser.add_argument("--model", default='pretrained_models/modelComplete_2019-06-12 17:05:30')

    parser.add_argument("--rotation_aug", default=True)
    parser.add_argument("--saveImages", default=False)
    parser.add_argument("--saveImages_All", default=False)
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--track_file", default="world_traj_kitti_with_intervals_correct.json")
    return parser.parse_args()


def main(config):
    v = evaluate_model.Validator(config)
    print('start test')
    v.test_model()


if __name__ == "__main__":
    config = parse_config()
    main(config)
