# Multifactor Transformer Model

- File Structure:
  - FYP_model:
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

- workflow -> preprocessing -> preparator -> train -> eval
- What each file does and outputs
  - Backbone.py
    - Summary:  includes linear backbone class and convolution backbone class to project the dataset into a lower dimension dataset before feeding it to the pre-processor
    - Input: output dimensions, number of layers
    - Output: linear backbone class object and convolution backbone class object
  - Model.py
    - Summary: construct the layers of the model and compile with the specified loss function, scheduler, and optimizers
    - Inputs: input shape, multi-attention parameters, and scheduler boolean
    - Outputs: model object
  - Positional_encoding.py
    - Summary: Time2Vec class definition to produce encoded positional value for time-series data
    - Input: None
    - Output: Time2Vec class object
  - Transformer.py
    - Summary: Class definitions for the layers of the model which includes single attention layer, multi attention layer and transformer encoder.
    - Input: None
    - Output: single attention class object, multi attention class object, transformer encoder class object
  - Results
    - Screenshots of our model performance across different equities and different pillars
  - All_exp.ipynb
    - Summary: A jupyter notebook to download all the required technical data and combine it with macroeconomic, value, and fundamental pillars. After that, the data will go through preprocessing and PCA before being splitted into train, test, and validation dataset
    - Output: train, test, validation CSV files that contain model-ready dataset.
  - Eval.ipynb
    - Summary: A jupyter notebook that trains the model using the pre-processed dataset. The notebook also evaluates the trading performance of the trained model.
    - Output: A table that shows the evaluation metrics and a graph that shows the trading balance over the course of the trading period
  - Macro_exp.ipynb
    - Summary: A jupyter notebook to download all the required technical data and combine it with macroeconomic pillars. After that, the data will go through preprocessing and PCA before being splitted into train, test, and validation dataset
    - Output: train, test, validation CSV files that contain model-ready dataset.
  - Preparator.py
    - Summary: Prepares the training data by loading the data and splitting the data in training, validation, and testing data 
    - Input: None
    - Output: Returns the data now split into train, validation, and test 
  - Tech_exp.ipynb
    - Summary: A jupyter notebook to download all the required technical data. After that, the data will go through preprocessing and PCA before being splitted into train, test, and validation dataset
    - Output: train, test, validation CSV files that contain model-ready dataset.
  - train.py:
    - Summary: Imports the transformer model from model.py then creates and runs the model-based the inputs provided.
    - Inputs: ticker, pillar
    - Outputs: Saves an instance of the generated model and the model progress
  - Transformer_model.ipynb:
    - Summary: A jupyter notebook that shows the whole process for one ticker from loading the data, preprocessing, split into train, test, and validation dataset, running the model, and showcasing the model output.
    - Input: ticker data file
    - Output: Charts of the predicted closing returns of Training, Test, and validation against the actual closing returns, as well as the Model loss, MAE, and MAPE charts
  - Val_exp.ipynb
    - Summary: A jupyter notebook to download all the required technical data and combine it with value pillars. After that, the data will go through preprocessing and PCA before being splitted into train, test, and validation dataset
    - Output: train, test, validation CSV files that contain model-ready dataset.


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
