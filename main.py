from config import get_configuration
from experiment import write_results_to_file, load_data, compare_models
from pypots.imputation import GRUD, BRITS, mrnn, gpvae, usgan, CSDI, TimesNet, SAITS
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_configuration()

    # 定义模型字典
    models = {
        "GRU-D": GRUD,
        "BRITS": BRITS,
        "M-RNN": mrnn.MRNN,
        "GP-VAE": gpvae.GPVAE,
        "US-GAN": usgan.USGAN,
        "CSDI": CSDI,
        "TimesNet": TimesNet,
        "SAITS": SAITS
    }

    # 参数配置字典
    parameters = {
        "GRU-D": {"rnn_hidden_size": 32},
        "BRITS": {"rnn_hidden_size": 32},
        "M-RNN": {"rnn_hidden_size": 32},
        "GP-VAE": {"latent_size": 64},
        "US-GAN": {"rnn_hidden_size": 32},
        "CSDI": {"n_layers": 4, "n_heads": 4, "n_channels": 64, "d_time_embedding": 32,
                 "d_feature_embedding": 32, "d_diffusion_embedding": 32},
        "TimesNet": {"n_layers": 2, "top_k": 16, "d_model": 128, "d_ffn": 128,
                     "n_kernels": 2},
        "SAITS": {"n_layers": 4, "d_model": 128, "n_heads": 4, "d_k": 32, "d_v": 16,
                  "d_ffn": 16}
    }

    # 加载数据
    data = load_data(file_path='wind_0001_1h_11k.csv', columns=['windSpeed2m', 'windSpeed10m'])

    # 比较模型
    results = compare_models(data, models, parameters, epochs=args.epochs, batch_size=args.batch_size, device=device,
                             missing_mode='continuous', missing_rate=args.missing_rate, n_features=2,
                             n_steps=args.seq_length, patience=1,
                             max_missing_length=args.max_missing_rate * args.missing_rate * 1160)

    # 打印结果
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAE: {metrics['MAE']:.3f}")

    # 写入结果到文件
    file_path = 'results.txt'
    write_results_to_file(results, file_path)


if __name__ == "__main__":
    main()
