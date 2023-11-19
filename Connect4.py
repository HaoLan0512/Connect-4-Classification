import numpy as np
import random
import pandas as pd

def empty_columns(A):
    empty = (A[0,:]==0)
    return [i for i in range(len(empty)) if empty[i]]

def check_winning(A,connectcount):
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
                elif np.allclose(A[i,j:j+connectcount],np.ones(connectcount)):
                    return 2
    return 0

def generate_policy(A,turn,connectcount=4):
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
    policy = random.randint(0,len(empty_indices)-1)
    [A,turn] = generate_boardstate(A, turn, policy)
    return [A,turn]

def generate_boardstate(A,turn,policy):
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
    A = np.zeros((rows,columns))
    turn = 1
    for i in range(0,turncount):
        [A,turn]=generate_policy(A,turn)
    return [A,turn]


[A,turn]=generate_random_boardstate(10)
print(np.reshape(A, (1,42))[0])
print(turn)

dataset = pd.read_csv('connect-4.csv')
dataset = np.array(dataset)
first_player = (dataset[0]=='x').astype(int)
second_player = 2*(dataset[0]=='o').astype(int)
d = np.reshape((first_player+second_player)[0:42],(7,6)).T
print(d)
print(dataset[0])