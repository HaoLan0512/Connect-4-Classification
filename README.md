# Connect-4-Classification-and-Prediction
Math 5465 Final Project: using classification to predict the winner of Connect 4 for a given board state.

## Data sources and references
http://archive.ics.uci.edu/dataset/26/connect+4 \

https://tromp.github.io/c4/c4.html

## Role of each file
**connect-4.csv**: Raw data download from the above link. \
\
**CleanedData.csv**: Data used in Connect 4 KNN.ipynb and ConnectFour_SVM_12_11_23.py. \
\
**CleanedData _pytorch.csv**: Data used in Connect 4 Torch.ipynb and Connect 4 Boardstate Prediction with Neural Network .ipynb (change the label from above). \
\
**Connect 4 KNN.ipynb**: Use KNN to classify. \
\
**ConnectFour_SVM_12_11_23.py**: Use SVM and kernel SVM to classify. \
\
**Connect 4 Torch.ipynb**: Use pytroch to train the neural network. \
\
**Connect_4_Classification.pth**: Model saved after training and loaded in Connect 4 Boardstate Prediction with Neural Network .ipynb.\
\
**Connect 4 Boardstate Prediction with Neural Network .ipynb**: To play with this, please download download CleanedData _pytorch.csv and Connect_4_Classification.pth. It allows the user to enter a board state, and if the user chooses to leave it blank, it will randomly generate a board state containing the number of rounds the user now wants. After that, it will use the model trained above to predict the next best move.
