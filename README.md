# Multifactor Transformer Model
## Versions
- v0.0

- file structure
  - testing
  - factor selection
- workflow -> preprocessing -> preparator -> train -> eval
- What each file does and outputs
  - train.py: 
    - Summary: Imports the transformer model from model.py then creates and runs the model based the inputs provided.
    - Inputs: ticker, pillar
    - Outputs: Saves an instance of the generated model and the model progress 

## File structure
* model
  * backbone.py: includes linear backbone class and convolution backbone class to project the dataset into a lower dimension dataset before feeding it to the pre-processor
  * model.py
  * positional_encoding.py
  * transformer.py
* results
  * all
    * AAPL.jpg
    * AMZN.jpg
    * MSFT.jpg
  * macro
    * AAPL.jpg
    * AMZN.jpg
    * MSFT.jpg
    * SPY.jpg
  * value
    * AAPL.jpg
    * AMZN.jpg
    * MSFT.jpg
* all_exp.ipynb
* eval.ipynb
* macro_exp.ipynb
* main.py
* preparator.py
* tech_exp.ipynb
* train.py
* transformer_model.ipynb
* val_exp.ipynb
