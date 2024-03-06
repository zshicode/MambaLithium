# MambaLithium: Selective state space model for state-of-charge estimation of lithium-ion batteries

Lithium-ion batteries is crucial in electric vehicles and new energy industry. State-of-charge (SOC) are one of the key states of lithium-ion batteries. As Mamba (Structured state space sequence models with selection mechanism and scan module, S6) has achieved remarkable success in sequence modeling tasks, this repository proposes a Mamba-based model to predict the SOC. The design of the model is similar to MambaStock[https://github.com/zshicode/MambaStock] (see Citation).

## Requirements

The code has been tested running under Python 3.7.4, with the following packages and their dependencies installed:
```
numpy==1.16.5
matplotlib==3.1.0
sklearn==0.21.3
pandas==0.25.1
pytorch==1.7.1
```

The data used in this repository was downloaded from https://github.com/GuoKent/Hybrid_time_series_forecasting_model. Some code of the Mamba model is from https://github.com/alxndrTL/mamba.py

## Usage

The `./data` folder includes three datasets: DST, FUDS, US06. Following previous research (Yang et al., 2019; Chen et al., 2024), either of these datasets can be testing set, and the corresponding other two datasets are used for training. Users can choose SOC measured under different temperature (Celsius degree: 0C, 10C, 25C, 30C, 40C, 50C) for prediction.

```
python main.py --epochs 1000
```

## Options

We adopt an argument parser by package  `argparse` in Python, and the options for running code are defined as follow:

```python
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2,
                    help='Num of layers')
parser.add_argument('--test', type=str, default='FUDS',
                    help='Test set')
parser.add_argument('--temp', type=str, default='25',
                    help='Temperature')                    

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
```

## References

Yang et al., State-of-Charge Estimation of Lithium-Ion Batteries via Long Short-Term Memory Network, IEEE Access, 2019

Chen et al., An LSTM-SA model for SOC estimation of lithium-ion batteries under various temperatures and aging levels, J Energy Storage, 2024

## Citation

```
@article{shi2024mamba,
  title={MambaStock: Selective state space model for stock prediction},
  author={Zhuangwei Shi},
  journal={arXiv preprint arXiv:2402.18959},
  year={2024},
}
```