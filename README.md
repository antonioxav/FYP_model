# Multifactor Transformer Model

- File Structure:
  ```lua
  .
  |   .gitignore
  |   all_exp.ipynb
  |   eval.ipynb
  |   Factor_Selection.ipynb
  |   fund_exp.ipynb
  |   LICENSE
  |   macro_exp.ipynb
  |   main.py
  |   preparator.py
  |   README.md
  |   tech2_exp.ipynb
  |   tech_exp.ipynb
  |   train.py
  |   val_exp.ipynb
  |   __init__.py
  |   
  +---data
  |   +---processed
  |   |   +---all
  |   |   |       .gitignore
  |   |   |
  |   |   +---fund
  |   |   |       .gitignore
  |   |   |
  |   |   +---macro
  |   |   |       .gitignore
  |   |   |
  |   |   +---tech
  |   |   |       .gitignore
  |   |   |
  |   |   +---test
  |   |   |       .gitignore
  |   |   |
  |   |   \---value
  |   |           .gitignore
  |   |
  |   \---raw
  |           .gitignore
  |
  +---instances
  |       .gitignore
  |
  +---logs
  |       .gitignore
  |
  +---model
  |       backbone.py
  |       model.py
  |       positional_encoding.py
  |       transformer.py
  |       __init__.py
  |
  +---results
  |   +---all
  |   |       AAPL.jpg
  |   |       AMZN.jpg
  |   |       MSFT.jpg
  |   |
  |   +---macro
  |   |       AAPL.jpg
  |   |       AMZN.jpg
  |   |       MSFT.jpg
  |   |       SPY.jpg
  |   |       
  |   \---value
  |           AAPL.jpg
  |           AMZN.jpg
  |           MSFT.jpg
  |
  +---tests
  |       self_generated_data_set_1.py
  |       self_generated_data_set_2.py
  |
  \---utils
          plot_utils.py
          preprocessing_utils.py
          __init__.py
  ```

## Workflow 
- Pre-processing -> Preparator -> Train -> Evaluation
- Pre-processing:
  - Use the individual {pillar name}_exp.ipynb files to do the data pre-processing for the chosen factor individual or combined factor pillar model.
  - NOTE: Factor_Selection.ipynb can be referred to for the factor selection code and results.
- Preparator:
  - Use the preparator.py file to reshape the pre-processed data into a model-feedable shape.
- Train:
  - Use the train.py file for training the model itself for the selected pillar or combined.
- Evaluation:
  - Use the eval.ipynb to view the model results.