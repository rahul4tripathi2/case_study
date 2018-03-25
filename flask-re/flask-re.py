import flask
from flask import Flask, request
import json
import CollaborativeFiltering2
from CollaborativeFiltering2 import load2,loadPeerFile
import CollaborativeFiltering
from CollaborativeFiltering import load1, loadItemFile, similar
import ContentBasedFiltering
from ContentBasedFiltering import loadUserFile,loadUserData
from ContentBasedFiltering import generate_recommendation
import pickle
import os

print "******************************"
print "Reading Environment Variables"
WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])
WFTP_DATA_HOME = os.path.abspath(os.environ['WFTP_DATA_HOME'])
WFTP_SCRIPTS_HOME = os.path.abspath(os.environ['WFTP_SCRIPTS_HOME'])
WFTP_MODEL_HOME = os.path.abspath(os.environ['WFTP_MODEL_HOME'])
print "Reading Environment Variables is Completed"

print "******************************"

def getUser(param_1):
    try:
        os.chdir(WFTP_HOME)
        users = pickle.load(open('users.pkl', 'rb')) 
        print "global users pkl exist"
    except:
        print "global users pkl doesn't exist"   
        
    return users

app = flask.Flask(__name__)

@app.route("/api/aaip/python/recommend/peer", methods=['POST'])
def getRecommendationPeer():
    print('Recommendations')
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    user = input_query.get("User")
    num = input_query.get("No of Recommendations")
    users = getUser(param_1)
    User_matrix_Revised, UserCompletionList = CollaborativeFiltering2.loadPeerFile(param_1)
       
    if str(user) in users[0].tolist():
        (start_time,output,end_time) = CollaborativeFiltering2.generate_recommendation(user,User_matrix_Revised,UserCompletionList,num)
        print("---Peerbased Recommendation is completed in %s seconds ---" % (end_time - start_time))
        list1 = str(list(output.index[0:len(output)]))
        response = app.response_class(response=json.dumps(list1),
                              status=200, mimetype='application/json') 
        return response
    else:
        return 'No Recommendation as there is no history available'
        

@app.route("/api/aaip/python/recommend/item", methods=['POST'])
def getRecommendationItem():
    print('Recommendations')
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    user = input_query.get("User") 
    noofRecommendations = input_query.get("No of Recommendations")
    users = getUser(param_1)
    BoardRatings, similar_board, UserRatings_Item = CollaborativeFiltering.loadItemFile(param_1)
        
    if str(user) in users[0].tolist():
        (start_time,output,end_time) = CollaborativeFiltering.generate_recommendation(BoardRatings, similar_board, UserRatings_Item, user, noofRecommendations)
        print("---Itembased Recommendation is completed in %s seconds ---" % (end_time - start_time))
        list1 = str(list(output.index[0:len(output)]))
        response = app.response_class(response=json.dumps(list1),
                              status=200, mimetype='application/json') 
        return response
    else:
        return 'No Recommendation as there is no history available'
          
@app.route("/api/aaip/python/recommend/user", methods=['POST'])
def RecommendationsUser():
    print('Starting Recommendations tasks for Usecase1')
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    param_2 = input_query.get("User")
    param_3 = input_query.get("No of Recommendations")
    
    UserBoardVector, LearningBoardVector, CompletionSeries,users = ContentBasedFiltering.loadUserFile(param_1)	 
           
    if str(param_2) in list(users):
        start_time,output,end_time = generate_recommendation(UserBoardVector, LearningBoardVector, CompletionSeries,param_2,param_3)
        print("---Userbased Recommendation is completed in %s seconds ---" % (end_time - start_time))
        list2 = str(list(output.index[0:len(output)]))
        response = app.response_class(response=json.dumps(list2),
                              status=200, mimetype='application/json')
        return response                      
    else:
        return 'No Recommendation as there is no history available'
        
@app.route("/api/aaip/python/train/peer", methods=['POST'])
def getTrainingPeer():
    print('Training')
    os.chdir(WFTP_DATA_HOME)
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    param_2 = input_query.get("User_Detail")
    param_3 = input_query.get("Completion_Board")
    os.chdir(WFTP_DATA_HOME + '\\'+ param_1)
    print "Training is completed please wait for pickling to be done"
       
    start_time,User_matrix_Revised,UserCompletionList,end_time = load2(param_2,param_3,param_1)
    
        
    os.chdir(WFTP_MODEL_HOME + '\\' + param_1)
    User_matrix_Revised.to_pickle('User_matrix_Revised.pkl')
    UserCompletionList.to_pickle('UserCompletionList.pkl')
    print "Pickling Done"
    print("---Peerbased Training is completed in %s seconds ---" % (end_time - start_time))
    
    return 'Peer Training is Completed and Model has been saved to WFTP model directory'
    
          
@app.route("/api/aaip/python/train/item", methods=['POST'])
def getTrainingItem():
    print('Training')
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    param_2 = input_query.get("Completion_Board")
    os.chdir(WFTP_DATA_HOME + '\\'+ param_1)
    
    start_time,UserRatings_Item,end_time = load1(param_2,param_1)
    print("---Itembased Training is completed in %s seconds ---" % (end_time - start_time))
      
    print "Training is completed please wait for pickling to be done"
        
    os.chdir(WFTP_MODEL_HOME + '\\' + param_1)
    BoardRatings,similar_board = similar(UserRatings_Item)
    UserRatings_Item.to_pickle('UserRatings_Item.pkl')
    BoardRatings.to_pickle('BoardRatings.pkl')
    similar_board.to_pickle('similar_board.pkl') 
    print "Pickling Done"
    return 'ItemBased Training is Completed and Model has been saved to WFTP model directory'
    
@app.route("/api/aaip/python/train/user", methods=['POST'])
def TrainingUser():
    print('Started Training for Usecase1')
    input_query = request.get_json()
    param_1 = input_query.get("ClientID")
    param_2 = input_query.get("BoardWith_Tags")
    param_3 = input_query.get("Completion_Board")
    param_4 = input_query.get("Endorse")
    param_5 = input_query.get("Follow")
    param_6 = input_query.get("Likes")
    param_7 = input_query.get("Dummy")
    
    os.chdir(WFTP_DATA_HOME + '\\' + param_1) 
    
    (start_time,UserBoardVector,LearningBoardVector,CompletionSeries,end_time) = loadUserData(param_2,param_3, param_4, param_5, param_6,param_7) 
    os.chdir(WFTP_MODEL_HOME + '\\' + param_1)
    
    print("Saving Pickle Files for usecase1")
    
    UserBoardVector.to_pickle('UserBoardVector.pkl')
    LearningBoardVector.to_pickle('LearningBoardVector.pkl') 
    CompletionSeries.to_pickle('CompletionSeries.pkl')
    print("---Userbased Training is completed in %s seconds ---" % (end_time - start_time))
    print 'Training Completed, Model saved to Client Model directory, please stop the code execution'
    return 'User Training is Completed and Model has been saved to respective client model directory'
    
    
if __name__ == "__main__":
    app.run()

