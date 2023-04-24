from train import train
from preparator import prepare


def main():
    ticker = 'MSFT'
    pillar = 'macro'
    pca = 'all'
    task = 'reg'
    seq_len = 10

    experiment_name = f'{ticker}_{pillar}_reg10_4_pca_all_huber0.1'

    prepare(ticker, pillar, pillar, seq_len, pca, task)
    train(ticker, pillar, pca, experiment_name)

if __name__=='__main__':
    main()


