import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random

#Packages needed for KNN
from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter

def empty_columns(A):
    """Test whether a given boardstate has legal move
    ======
    Find all columns with 0 at the top

    Parameters
    ----------
    A : numpy array, int
        6*7 matrix, represent a given board state: 1 means Player 1, 2 means Player2, 0 means empty.

    Returns
    -------
    empty_indices : list, int
        Index of columns that are not full.
    """
    empty = (A[0,:]==0)
    return [i for i in range(len(empty)) if empty[i]]


def check_winning(A,connectcount):
    """Test whether a given boardstate has a winner
    ======
    Find #1 or #2 in the given matrix reaches the given connectcount value.

    Parameters
    ----------
    A : numpy array, int
        6*7 matrix, represent a given board state: 1 means Player 1, 2 means Player2, 0 means empty.
    connectcount : int
        # of 1 or 2 in a line required for win.

    Returns
    -------
    w: int
    1 : Player 1 win.
    2 : Player 2 win.
    0 : There is no winner in the current boardstate.
    """
    for i in range(np.size(A,0)):
        for j in range(np.size(A,1)):
            if A[i,j]==0: #No need to waste time if the spot is empty
                continue

            if i<(np.size(A,0)-connectcount+1): #That is, if we are far enough from the bottom for a connect-X
                #Check columns below i,j
                if np.allclose(A[i:i+connectcount,j],np.ones(connectcount)):
                    return 1
                elif np.allclose(A[i:i+connectcount,j],(np.ones(connectcount)*2)):
                    return 2

                #Check down-right diagonals, if able
                if j<(np.size(A,1)-connectcount+1):
                    if np.allclose(np.diag(A[i:i+connectcount,j:j+connectcount]),np.ones(connectcount)):
                        return 1
                    elif np.allclose(np.diag(A[i:i+connectcount,j:j+connectcount]),(np.ones(connectcount)*2)):
                        return 2
                elif j>(connectcount-2): #Check down-left diagonals, if able
                    if np.allclose(np.diag(np.fliplr(A[i:i+connectcount,j-connectcount+1:j+1])),np.ones(connectcount)):
                        return 1
                    elif np.allclose(np.diag(np.fliplr(A[i:i+connectcount,j-connectcount+1:j+1])),(np.ones(connectcount)*2)):
                        return 2

            if j<(np.size(A,1)-connectcount+1): #Check rows to the right if able
                if np.allclose(A[i,j:j+connectcount],np.ones(connectcount)):
                    return 1
                elif np.allclose(A[i,j:j+connectcount],(np.ones(connectcount)*2)):
                    return 2
    return 0
     

def generate_policy(A,turn,connectcount=4):
    """Randomly pick an available column to make the next move and determine whose turn next.
    ======
    Use the # of turn to determine the game state. If the game doesn't end, then generate a random move and switch turns.

    Parameters
    ----------
    A : numpy array, int
        6*7 matrix, represent a given board state: 1 means Player 1, 2 means Player2, 0 means empty.
    connectcount : int
        # of 1 or 2 in a line required for win.
    turn: int
        Represent the game state: 1, player 1 take the next move; 2, player 2 take the next move; 3, player 1 win;
        4, player 2 win; 5, a tie, i.e. no avaiable move but no one wins.
    policy: int
        A randomly generated integer represent the index of the next move column.

    Returns
    -------
    [A,turn]: list,
    A: 6*7 matrix, boardstate updated; turn: int, next turn updated.
    """
    if turn>2:
      print("Game is over!")
      if turn==3:
            print("Player 1 wins")
      elif turn==4:
            print("Player 2 wins")
      elif turn==5:
            print("It's a tie!")
      return [A,turn]
    w = check_winning(A,connectcount)
    if w==1:
        print("Game is over!")
        print("Player 1 wins")
        return [A,3]
    if w==2:
        print("Game is over!")
        print("Player 2 wins")
        return [A,4]
    empty_indices = empty_columns(A)
    if len(empty_indices)==0:
        print("Game is over!")
        print("It's a tie!")
        return [A,5]
    policy = empty_indices[random.randint(0,len(empty_indices)-1)]
    [A,turn] = generate_boardstate(A, turn, policy)
    return [A,turn]

def generate_boardstate(A,turn,policy):
    """Generate the next boardstate.
    ======
    Use policy to update the boardstate and switich turn.

    Parameters
    ----------
    A : numpy array, int
        6*7 matrix, represent a given board state: 1 means Player 1, 2 means Player2, 0 means empty.
    turn: int
        Represent the game state: 1, player 1 take the next move; 2, player 2 take the next move; 3, player 1 win;
        4, player 2 win; 5, a tie, i.e. no avaiable move but no one wins.
    policy: int
        A randomly generated integer represent the index of the next move column.

    Returns
    -------
    [A,turn]: list,
    A: 6*7 matrix, boardstate updated; turn: int, next turn updated.
    """
    if turn<3:
        m = np.size(A,0)
        for i in range(m):
            if A[m-i-1,policy]==0:
                A[m-i-1,policy]=turn
                if turn==1:
                    turn=2
                elif turn==2:
                    turn=1
                return [A,turn]
        return "Error - Bad Policy choice!"
    return "The game is already over! Turn count too high"

def generate_random_boardstate(turncount,rows=6,columns=7):
    """Randomly simulate the first n move of a empty boardstate.
    ======

    Parameters
    ----------
    turncount: int
    # of steps required to simulate.

    Returns
    -------
    [A,turn]: list,
    A: 6*7 matrix, boardstate updated; turn: int, next turn updated.
    """

    restart = True

    while restart:

        restart = False
        A = np.zeros((rows,columns))
        turn = 1

        for i in range(0,turncount):
            [A,turn]=generate_policy(A,turn)
            if check_winning(A,4)!=0:
                restart = True
                break
    return [A,turn]

def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))

def find_neighbors(distances, k):
    return distances[0:k]

def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))

def knn(training_set, x_test, y_test, k):
    distances = []
    dist = 0
    limit = len(training_set[0]) - 1
    label_predict = []

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in x_test:
            for row in training_set:
                for x, y in zip(row[:limit], test_instance):
                    dist += (x-y) * (x-y)
                distances.append(row + [sqrt(dist)])
                dist = 0

            distances.sort(key=itemgetter(len(distances[0])-1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)
            label_predict.append(classes[index])
            # print(label_predict)

            # Display prediction
            # print('The predicted class for sample ' + str(test_instance) + ' is : ' + str(classes[index]))
            # print('Number of votes : ' + str(value) + ' out of ' + str(k))

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)
    
    #Testing accuracy
    test_acc = np.mean(label_predict == y_test)*100
    # print(label_predict)
    # print('Testing Accuracy: %.2f%%'%test_acc)

    return test_acc
    
def svm(x_train,x_test,y_train,y_test):
    distances = []
    dist = 0
    label_predict = []
    
    #Train SVM
    clf = SVC(kernel='poly')
    clf.fit(x_train,y_train)
    
    #Training accuracy
    y_pred = clf.predict(x_train)
    train_acc = np.mean(y_pred == y_train)*100
    print('Training Accuracy: %.2f%%'%train_acc)
    
    #Testing accuracy
    y_pred = clf.predict(x_test)
    test_acc = np.mean(y_pred == y_test)*100
    print('Testing Accuracy: %.2f%%'%test_acc)
    
    return [train_acc,test_acc]


#a = [65.41221853413475, 65.72620435991746, 65.84282766663677, 66.12092939804431, 65.73517538351126, 65.44361711671301, 66.2420382165605, 65.72171884812057, 65.76208845429264, 65.77554498968333, 65.94599443796537]
#print(np.mean([65.41221853413475, 65.72620435991746, 65.84282766663677, 66.12092939804431, 65.73517538351126, 65.44361711671301, 66.2420382165605, 65.72171884812057, 65.76208845429264, 65.77554498968333, 65.94599443796537]))

dataset = pd.read_csv('CleanedData.csv')
dataset = np.array(dataset)

data = dataset[:, :-1]
target = dataset[:, -1]
accuracy_list =[]
loop=0
for random_seed in range(40,51):
    loop+=1
    print("Loop: ",loop)
    x_train, x_test, y_train, y_test = train_test_split(data[:300], target[:300], test_size=0.33,random_state=random_seed)
    train_set = np.column_stack((x_train, y_train)).tolist()
    [train_acc,test_acc] = svm(x_train,x_test,y_train,y_test)
    accuracy_list.append(test_acc)

print(accuracy_list)
print(np.mean(accuracy_list))