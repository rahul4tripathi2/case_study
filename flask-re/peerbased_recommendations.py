import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pickle
import os
import time
from sklearn.preprocessing import LabelEncoder



WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])


def load_peer_pickle(client_id):
    os.chdir(WFTP_HOME + '\\Model' + '\\'+ client_id)
    try:
        user_matrix_revised = pickle.load(open('User_matrix_Revised.pkl', 'rb'))
        user_completion_list = pickle.load(open('UserCompletionList.pkl', 'rb'))
        print ('Existing model for Usecase2 Userbased will be used for recommendation') 
    except:
        print ('The model files for Usecase2 UserBased are missing , need to train the recommendation engine or put the model files in WFTP Model Directory')
    
    return user_matrix_revised, user_completion_list
       
def data_prepocessing(filename1, client_id):
    os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
    user_detail = pd.read_csv(filename1)
    user_detail['emp_level'] = user_detail['emp_level'].str.lower()
    le = LabelEncoder()
    user_detail["emp_level"] = le.fit_transform(user_detail["emp_level"])
    os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
    user_detail.to_pickle('user_detail_encode.pkl')
    users = pickle.load(open('user_detail_encode.pkl', 'rb'))
    return users

def load_peer_csv(filename1,filename2,location):
    start = time.time()
    users = get_user(filename2,location)
    users = pd.DataFrame(users)
    os.chdir(WFTP_HOME + '\\Model' +'\\' + location)
    users.to_pickle('users.pkl')
    os.chdir(WFTP_HOME + '\\Data' +'\\' + location)
    users = data_prepocessing(filename1, location)
    users = users.rename(columns = {"enterpriseid": "enterprise_id", "emp_level": "emp_designation"})
    user_vector = pd.concat([users[['enterprise_id','emp_designation']],pd.get_dummies(users.gu), pd.get_dummies(users.workforcegroup), pd.get_dummies(users.orglevel2desc)], axis = 1)
    dist_matrix = pd.DataFrame(squareform(pdist(user_vector.ix[:, 1:])), columns=user_vector.enterprise_id.unique(), index=user_vector.enterprise_id.unique())
    dist_matrix[dist_matrix > 0] = dist_matrix + 10
    dist_matrix[dist_matrix == 0] = 1
    dist_matrix[dist_matrix >= 10] = 0
    os.chdir(WFTP_HOME + '\\Data' + '\\' + location)
    user_completion = pd.read_csv(filename2)
    user_completion = user_completion.rename(columns = {"enterprise_ID": "enterprise_id", "Board_title": "board_title","Board_Id": "board_Id"})
    cols_df = ['enterprise_id','board_title','board_Id']
    user_completion = user_completion[cols_df]
    user_completion = user_completion.drop_duplicates()
    user_completion['Status'] = 1

    user_completion['combined'] = user_completion['board_Id'].map(str)
    user_completion_pivot = user_completion.pivot_table(index = ['board_title'], columns = ['enterprise_id'], values = 'Status', fill_value = 0)
    user_completion_corr = user_completion_pivot.corr()
    user_matrix_revised = dist_matrix * user_completion_corr
       
    user_completion_list = pd.Series(data = list(user_completion['combined']), index = list(user_completion['enterprise_id']))
    end = time.time()
    return start,user_matrix_revised,user_completion_list,end
    
    
def get_user(filename2,location):
    os.chdir(WFTP_HOME + '\\Data' + '\\'+ location)
    user_completion = pd.read_csv(filename2)
    users = user_completion.enterprise_ID.unique()
    return users

def generate_recommendation(user,user_matrix_revised,user_completion_list,num):
    start = time.time()
    simboard = pd.Series()
    L = user_matrix_revised.loc[user]
    for username in set(user_completion_list.index.sort_values()):
        similarity = L[username]
        if similarity > 0 and similarity <= 1:
            completed_courses = pd.Series(user_completion_list[username])
            completed_courses_ratings = pd.Series(index = completed_courses.values, data = np.repeat(1,len(completed_courses)))
            sims = completed_courses_ratings.map(lambda x:x  * similarity)
            simboard = simboard.append(sims)
    simboard = simboard.groupby(simboard.index).sum()
    completed_courses = pd.Series(user_completion_list[user])
    filtered_sims = simboard.drop(completed_courses.values)
    if num:
        result = filtered_sims.sort_values(ascending=False).head(num)
        end = time.time() 

    else:
        result = filtered_sims.sort_values(ascending=False).head(10)
        end = time.time() 

    return start,result,end
   


