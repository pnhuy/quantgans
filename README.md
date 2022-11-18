# Quant GANs

Student: Huy Pham

Email: nfam_2@edu.hse.ru

This repository demostrates the paper: [Wiese et al., Quant GANs: Deep Generation of Financial Time Series, 2019](https://arxiv.org/pdf/1907.06673.pdf)

Please the notebook [QuantGans][QuantGANs.ipynb] for detail of the model.

## Install the requirements

```
$ pip install -r requirements.txt
```

## Train

```
$ python train.py --data_path sample/sp500.csv
```

## Inference

```
$ python inference.py
```

## Results

### The training loss

![](logs/training_loss.png)

### The cumalative log return

![](logs/cumalative_log_return.png)

### The comparison between Real vs Synthetic lag

![](logs/real_vs_synthetic_lag.png)

### The comparison between Real vs Synthetic distribution

![](logs/real_vs_synthetic_dist.png)



