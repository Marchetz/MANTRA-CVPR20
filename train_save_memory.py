import argparse
import memory_writer


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

    parser.add_argument("--rotation_aug", default=True)
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--track_file", default="world_traj_kitti_with_intervals_correct.json")
    return parser.parse_args()


def main(config):
    print('Start writing into memory')
    s = memory_writer.MemoryWriter(config)
    s.mem_write()


if __name__ == "__main__":
    config = parse_config()
    main(config)
