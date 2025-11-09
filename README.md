Vanilla RNN on Text Sequences for Sentiment Classification
This notebook demonstrates the application of a Vanilla Recurrent Neural Network (RNN) using keras.layers.SimpleRNN for a basic sentiment classification task on short text sequences.

Thanks to Habib Sheikh for his learning inputs!

This project focuses on building and understanding a simple RNN for classifying text sentiment (positive/negative). It covers the entire pipeline from data preparation (tokenization, padding) to model training, evaluation, and visualization of internal RNN states.

What You'll Learn
How to build a toy text classification dataset with positive/negative sentiment.
The process of tokenizing text into integer sequences and inspecting the word-index mapping.
Techniques for padding sequences to a fixed length suitable for RNN input.
Constructing a Vanilla RNN model using Keras: Embedding → SimpleRNN → Dense.
Training, evaluating, and testing the model on new, unseen sentences.
Visualizing per-timestep outputs of the RNN with return_sequences=True to gain intuition about its internal workings.
How to Run
Open in Google Colab: Click the "Open in Colab" badge (if available) or upload the .ipynb file to your Google Drive and open it with Colab.
Install Dependencies: The notebook uses keras-preprocessing, which is installed via !pip install keras-preprocessing in the first code cell.
Run All Cells: Execute all cells sequentially (Runtime > Run all).
Dataset
The notebook utilizes a small, self-contained dataset of a few dozen short sentences manually labeled as either positive (1) or negative (0). This intentional simplicity allows for clear observation of tokenization and sequence processing.

Model Architecture
The Keras model employs the following layers:

Embedding Layer: Converts integer-encoded words into dense, fixed-size vectors (learned during training).
SimpleRNN Layer: The core recurrent layer, processing sequences one timestep at a time. By default, it returns the final hidden state for classification.
Dense Layer: A single output neuron with a sigmoid activation for binary classification (predicting the probability of positive sentiment).
Example Results
The trained model can predict sentiment for new sentences. Here's an example of its output:

'I absolutely loved the film'                 -> positive probability: 0.487
'The movie was not good'                      -> positive probability: 0.557
'What a fantastic performance'                -> positive probability: 0.494
'Boring and weak direction'                   -> positive probability: 0.383
'Happy ending and great acting'               -> positive probability: 0.392
'Terrible and depressing story'               -> positive probability: 0.358
(Note: Probabilities are illustrative and depend on model training and specific input.)
