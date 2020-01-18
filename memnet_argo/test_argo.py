import argparse
#from evaluate import evaluate_argo
from evaluate import evaluate_argo_map


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--preds_scene", type=int, default=20)
    parser.add_argument("--preds", type=int, default=20)

    #model
    parser.add_argument("--model", default='pretrained_models/controller/model2020-01-17 11:02:00')

    #model map
    #parser.add_argument("--model_pretrained", default='pretrained_models/argo/model_epoch_59_2020-01-10 16:34:59_MAP')
    # parser.add_argument("--model_pretrained", default='pretrained_models/argo/model_epoch_0_2020-01-14 11:08:06')
    # parser.add_argument("--model_controller", default='pretrained_models/argo/model2020-01-09 16:05:09_CONTROLLER')

    parser.add_argument("--visualize_dataset", default=False)
    parser.add_argument("--memory_saved", default=True)

    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--th_memory", type=int, default=0.05)

    return parser.parse_args()

def main(config):
    v = evaluate_argo_map.Validator(config)
    print('start test')
    v.test_model()

if __name__ == "__main__":
    config = parse_config()
    main(config)



