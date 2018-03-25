import pandas as pd
import pickle
from scipy.spatial.distance import cosine
import os
import time

WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])

def load_user_pickle(client_id):
    os.chdir(WFTP_HOME + '\\Model'+ '\\'+ client_id)
    try:
        user_board_vector = pickle.load(open('UserBoardVector.pkl', 'rb'))
        learning_board_vector = pickle.load(open('LearningBoardVector.pkl', 'rb'))
        completion_series = pickle.load(open('CompletionSeries.pkl','rb'))
        users = completion_series.index.unique()
        print ('Existing model for Usecase1 will be used for recommendation')
    except:
        print ('The model files are missing,need to train the recommendation engine or put the model files in Python Home Directory')
    
    return user_board_vector, learning_board_vector,completion_series,users

def load_user_csv(file_board_with_tags, file_user_completions, file_endorse, file_follow, file_likes, file_dummy):
    start = time.time()
    data = pd.read_csv(file_board_with_tags, skipinitialspace=True)
    completions = pd.read_csv(file_user_completions)
    endorse = pd.read_csv(file_endorse)
    follow = pd.read_csv(file_follow)
    likes = pd.read_csv(file_likes)
    dummy_data  = pd.read_csv(file_dummy)
    
    dummy_data = dummy_data.rename(index = str, columns = {"Enterprise_ID": "enterprise_id", "BoardName": "board_title","BoardID": "board_Id"})
    
    
    colscomp = ['enterprise_ID', 'Board_title','Board_Id']
    completions = completions[colscomp]
    completions = completions.drop_duplicates()
    completions = completions.rename(index = str, columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})
    completions.board_Id = completions.board_Id.astype(str)
    completions = completions[completions.board_Id != '\N']
    completions = completions.dropna()
    completions.board_Id = pd.to_numeric(completions.board_Id)
    completions.board_Id = completions.board_Id.astype(int)
    completions = completions.append(dummy_data)


    cols_follow = ['enterprise_ID', 'Board_title','Board_Id']
    follow = follow[cols_follow]
    follow = follow.drop_duplicates()
    follow = follow.rename(index = str, columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})

    cols_endorse = ['enterprise_ID', 'Board_title','Board_Id']
    endorse = endorse[cols_endorse]
    endorse = endorse.drop_duplicates()
    endorse = endorse.rename(index = str, columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})

    cols_likes = ['enterprise_ID', 'Board_title','Board_Id']
    likes = likes[cols_likes]
    likes = likes.drop_duplicates()
    likes = likes.rename(index = str, columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})

    follow['Follow_Score'] = 1
    endorse['Endorse_Score'] = 1
    likes['Like_Score'] = 1

    data = data.rename(columns = {"BName": "board_title","BoardID": "board_Id"})
    data.Tag_names = data.Tag_names.str.lower()
    data.Tag_names = data['Tag_names'].str.strip()
    data.Tag_names = data.Tag_names.str.replace(" " , "")


    data.head()

    learning_board_vector = pd.concat([data[['board_Id','board_title']],data.Tag_names.str.get_dummies(sep =',')],axis = 1)
    learning_board_vector.shape

    learning_board_vector.index = learning_board_vector['board_title']
    learning_board_vector = learning_board_vector.drop('board_title',1)


    completions_follow_merge1 = pd.merge(completions, follow, on = ['enterprise_id','board_Id'], how = 'left')
    completions_follow_merge = pd.merge(completions_follow_merge1, endorse, on = ['enterprise_id','board_Id'], how = 'left')
    completions_follow_merge_likes = pd.merge(completions_follow_merge, likes, on = ['enterprise_id','board_Id'], how = 'left')
    
    cols = ['board_Id', 'enterprise_id', 'board_title_x','Follow_Score','Endorse_Score','Like_Score']


    
    completions_follow_merge_likes  = completions_follow_merge_likes [cols]
    completions_follow_merge_likes  = completions_follow_merge_likes.fillna(0)
    completions_follow_merge_likes['Score'] = completions_follow_merge_likes['Follow_Score'] + completions_follow_merge_likes['Endorse_Score']  + completions_follow_merge_likes['Like_Score'] + 1
    user_score = completions_follow_merge_likes
    merged_dataset = pd.merge(completions, data, on = 'board_Id', how = 'inner')
    completions.shape
    merged_dataset.shape
    
    user_score.BoardID = pd.to_numeric(user_score.board_Id)
  

    user_profile_matrix = pd.merge(merged_dataset,user_score, on = ['enterprise_id','board_Id'] )
    
    columns_for_user_profile_matrix = ['board_Id','enterprise_id','board_title_x_x','Tag_names','Score']
    user_profile_matrix = user_profile_matrix[columns_for_user_profile_matrix]
    
    
    user_board_vector = pd.concat([user_profile_matrix[['board_Id','enterprise_id','board_title_x_x','Score']],user_profile_matrix.Tag_names.str.get_dummies(sep =',')],axis = 1)

    user_board_vector = user_board_vector.drop('board_Id',1)
    user_board_vector = user_board_vector.drop('board_title_x_x',1)
    user_board_vector = user_board_vector.drop('Score',1)
    user_board_vector = user_board_vector.groupby(['enterprise_id']).sum()
    completion_series = pd.Series(data = list(completions.board_Id), index = [completions.enterprise_id] )

    
    total_columns_user_board_vector = user_board_vector.shape[1]
    total_columns_learn_board_vector = learning_board_vector.shape[1] - 1
    
    ubv = total_columns_user_board_vector
    lbv = total_columns_learn_board_vector
    cs = []
    print (total_columns_user_board_vector == total_columns_learn_board_vector)

    if total_columns_user_board_vector == total_columns_learn_board_vector:
        end = time.time()
        return start,user_board_vector, learning_board_vector, completion_series,end
    else:
         end = time.time()
         print ('Number of boards in Completions File and Learning Board with Tags is different, it should be same')
         return start,ubv, lbv, cs,end
         
 
def generate_recommendation(user_board_vector, learning_board_vector, completion_series,user,num):
    start = time.time()
    courses_completed = list(pd.Series(completion_series[user]).values)
    simboard = pd.Series()
    no_of_tags = learning_board_vector.shape[1]

    
    for var in range(0,len(learning_board_vector)):
        user_board_id = learning_board_vector.iloc[var,0]
        if user_board_id not in courses_completed:
            similarity = 1- cosine(user_board_vector.ix[user], learning_board_vector.iloc[var,1:no_of_tags])
            if similarity > 0:
                sims1 = pd.Series(data = similarity, index = [str(user_board_id)])
                simboard = simboard.append(sims1)

    if num:
        result = simboard.sort_values(ascending=False).head(num)
        end = time.time()
    else:
        result = simboard.sort_values(ascending=False).head(10)
        end = time.time()
        
    return start,result,end  
            
