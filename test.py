import argparse
import evaluate_MemNet

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=5)
    parser.add_argument("--model", default='pretrained_models/model_decoder_FT/model_FTdecoder2019-09-19 09:48:05')

    parser.add_argument("--visualize_dataset", default=False)
    parser.add_argument("--memory_saved", default=True)
    parser.add_argument("--withMRI", default=False)
    parser.add_argument("--saveImages_highlights", default=True)
    parser.add_argument("--saveImages_All", default=False)
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--track_file", default="world_traj_kitti_with_intervals_correct.json")
    return parser.parse_args()

def main(config):
    v = evaluate_MemNet.Validator(config)
    print('start evaluation')
    v.test_model()

if __name__ == "__main__":
    config = parse_config()
    main(config)



