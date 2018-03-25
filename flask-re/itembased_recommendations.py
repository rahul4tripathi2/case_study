import pandas as pd
from scipy.spatial.distance import cosine
import cPickle as pickle
import os
import time


WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])

def load_item_pickle(client_id):
    os.chdir(WFTP_HOME + '\\Model' + '\\'+ client_id)
    try:
        board_ratings = pickle.load(open('BoardRatings.pkl', 'rb'))
        similar_board = pickle.load(open('similar_board.pkl', 'rb')) 
        user_ratings_item = pickle.load(open('UserRatings_Item.pkl', 'rb'))
        print ('Existing model for usecase2 Itembased will be used for recommendation')
    except:
        print ('The model files for Usecase2 ItemBased are missing , need to train the recommendation engine or put the model files in WFTP Model Directory')
    
    return board_ratings, similar_board, user_ratings_item

    

def load_item_csv(filename, location):
    start = time.time()
    learning_board_completions = pd.read_csv(filename)
    learning_board_completions['Status'] = 1
    completed_courses = pd.DataFrame(learning_board_completions, columns = ['enterprise_ID','Board_title','Status','Board_Id'])
    completed_courses = completed_courses.rename(columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})
    completed_courses = completed_courses.drop_duplicates()
    users = completed_courses.enterprise_id.unique()
    completed_courses['board_Id'] = completed_courses.board_Id.astype(str)
    users = pd.DataFrame(users)
    os.chdir(WFTP_HOME + '\\Model' + '\\' + location)
    users.to_pickle('users.pkl')
    user_ratings_item = completed_courses.pivot_table(index=['enterprise_id'], columns =['board_Id'], values='Status',fill_value=0)
    end = time.time()
    return start,user_ratings_item,end
   
def similar(user_ratings_item):
    board_ratings = pd.DataFrame(index=user_ratings_item.columns,columns=user_ratings_item.columns)
    for i in range(0,len(board_ratings.columns)) :
        for j in range(0,len(board_ratings.columns)) :
            board_ratings.iloc[i,j] = 1-cosine(user_ratings_item.iloc[:,i],user_ratings_item.iloc[:,j])
            
    board_rating_no_of_cols = board_ratings.shape[1]
        
    similar_board = pd.DataFrame(index=board_ratings.columns,columns=range(1,board_rating_no_of_cols))
    for i in range(0,len(board_ratings.columns)):
        similar_board.iloc[i,:(board_rating_no_of_cols-1)] = board_ratings.iloc[0:,i].sort_values(ascending=False)[:(board_rating_no_of_cols-1)].index
        
    return board_ratings,similar_board


def item_similar_board(similar_board, board_id, no_of_board):
    start = time.time()
    board_id = str(board_id)

    board_id_values = similar_board.iloc[0].values

    if board_id not in board_id_values:
        result = []
        end = time.time()

    elif no_of_board:
        no_of_board += 1
        result = list(similar_board.loc[:board_id].iloc[:, 1:no_of_board].loc[board_id].values)
        if board_id in result:
            no_of_board += 1
            result = list(similar_board.loc[:board_id].iloc[:, 1:no_of_board].loc[board_id].values)
            result.remove(board_id)
        end = time.time()

    else:
        result = list(similar_board.loc[:board_id].iloc[:, 1:6].loc[board_id].values)
        if board_id in result:
            result = list(similar_board.loc[:board_id].iloc[:, 1:7].loc[board_id].values)
            result.remove(board_id)
        end = time.time()

    return start, result, end

def generate_recommendation(board_ratings,similar_board, user_ratings_item, user, no_of_recommendations):
    start = time.time()
    myratings = user_ratings_item.loc[user]
    simboard = pd.Series()
    for i in range(0, len(myratings.index)):
        sims = board_ratings[myratings.index[i]]
        sims = sims.map(lambda x: x * myratings[i])
        simboard = simboard.append(sims)
    
    simboard = simboard.groupby(simboard.index).sum()
    completed_courses = myratings[myratings > 0]
    filtered_sims = simboard.drop(completed_courses.index)
    filtered_sims = filtered_sims[filtered_sims.values > 0]

    if no_of_recommendations :
        result = filtered_sims.sort_values(ascending=False).head(no_of_recommendations)
        end = time.time()
    else :
        result = filtered_sims.sort_values(ascending=False).head(10)
        end = time.time()          
    return start,result,end





