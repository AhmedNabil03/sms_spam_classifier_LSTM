<h1>Text Classification with LSTM and GloVe Embeddings</h1>
This project implements a text classification model using LSTM (Long Short-Term Memory) neural networks with pre-trained GloVe word embeddings.<br>
The model is trained to classify text data into two categories based on a binary classification task.

<h2>Dataset</h2>
The dataset used for training and evaluation is sourced from kaggle "SMS spam dataset".<br>
It consists of 5572 Messages labeled as either "ham" or "spam".

<h2>Preprocessing</h2>
Before training the model, the text data undergoes several preprocessing steps including:<br>
<br>
Lowercasing text<br>
Removing HTML tags<br>
Removing special characters and numbers<br>
Removing stopwords<br>
Stemming words<be>

<h2>Model Architecture</h2>
The model architecture consists of an Embedding layer initialized with pre-trained GloVe word embeddings, followed by an LSTM layer for sequence processing and a Dense layer for classification. Dropout regularization is applied to prevent overfitting.

<h2>Training</h2>
The model is trained using a batch size of 32 for 20 epochs, with early stopping based on validation loss to prevent overfitting.<br>
The training and validation data are split with a ratio of 70:30.

<h2>Evaluation</h2>
The model's performance is evaluated based on accuracy and loss metrics on the validation set.<br>
Additionally, the final model's accuracy and loss metrics are printed at the end of training.

<h2>Requirements</h2>
Python<br>
TensorFlow<br>
NumPy<br>
Pandas<br>
NLTK<br>
Gensim
