# Multifactor Transformer Model
## Versions
- v0.0

- file structure
  - testing
  - factor selection
- workflow -> preprocessing -> preparator -> train -> eval
- What each file does and outputs

'''bash
|____preparator.py
|____val_exp.ipynb
|____LICENSE
|______init__.py
|____utils
| |______init__.py
| |____plot_utils.py
|____tech_exp.ipynb
|____README.md
|____results
| |____value
| | |____AAPL.jpg
| | |____AMZN.jpg
| | |____MSFT.jpg
| |____all
| | |____AAPL.jpg
| | |____AMZN.jpg
| | |____MSFT.jpg
| |____macro
| | |____AAPL.jpg
| | |____SPY.jpg
| | |____AMZN.jpg
| | |____MSFT.jpg
|____all_exp.ipynb
|____transformer_model.ipynb
|____.gitignore
|____macro_exp.ipynb
|____eval.ipynb
|____model
| |____backbone.py
| |______init__.py
| |____model.py
| |____transformer.py
| |____positional_encoding.py
|____train.py
|____.git
| |____config
| |____objects
| | |____pack
| | | |____pack-0303f3954b1514c8e10b0e2bcf6daa97ed558cdd.idx
| | | |____pack-0303f3954b1514c8e10b0e2bcf6daa97ed558cdd.pack
| | |____info
| |____HEAD
| |____info
| | |____exclude
| |____logs
| | |____HEAD
| | |____refs
| | | |____heads
| | | | |____main
| | | |____remotes
| | | | |____origin
| | | | | |____HEAD
| |____description
| |____hooks
| | |____commit-msg.sample
| | |____pre-rebase.sample
| | |____pre-commit.sample
| | |____applypatch-msg.sample
| | |____fsmonitor-watchman.sample
| | |____pre-receive.sample
| | |____prepare-commit-msg.sample
| | |____post-update.sample
| | |____pre-merge-commit.sample
| | |____pre-applypatch.sample
| | |____pre-push.sample
| | |____update.sample
| | |____push-to-checkout.sample
| |____refs
| | |____heads
| | | |____main
| | |____tags
| | |____remotes
| | | |____origin
| | | | |____HEAD
| |____index
| |____packed-refs
|____main.py
'''