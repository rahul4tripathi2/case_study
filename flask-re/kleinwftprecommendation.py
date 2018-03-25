from klein import run, route
import simplejson
import data_validation
from data_validation import validate_client_id,validate_csv, file_format_checker
import peerbased_recommendations
from peerbased_recommendations import load_peer_pickle,load_peer_csv
import itembased_recommendations
from itembased_recommendations import load_item_csv, load_item_pickle, similar, item_similar_board
import contentbased_recommendations
from contentbased_recommendations import load_user_pickle,load_user_csv
from contentbased_recommendations import generate_recommendation
import pickle
import os


WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])

def get_user(client_id):
    try:
        os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
        users = pickle.load(open('users.pkl', 'rb'))
        print ("global users pkl exist")
        return users
	  
    except:
        print ("global users pkl doesn't exist" )
        return ""
    

@route("/api/aaip/python/recommend/user", methods=['POST'])
def get_recommendation_user(request):
    print('starting user_based Recommendations')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        user = jsondata["User"]
        no_of_recommendations = jsondata["No of Recommendations"]
    except:
        dict = {"Error_Code": 701}
        response = simplejson.dumps(dict)
        return response


    errorcodelist = []

    if validate_client_id(client_id):
        dict = {"error_code": 703}
        errorcodelist.append(dict)
    else:
        user_board_vector, learning_board_vector, completion_series, users = contentbased_recommendations.load_user_pickle(client_id)
        if len(users) > 0 and str(user) not in list(users):
            dict = {"error_code": 702}
            errorcodelist.append(dict)

        if str(user) in list(users):
            start_time,output,end_time = generate_recommendation(user_board_vector, learning_board_vector, completion_series,user,no_of_recommendations)
            print("---User_based Recommendations is completed in %s seconds ---" % (end_time - start_time))
            list2 = str(list(output.index[0:len(output)]))
            dict = {"Board_id": list2, "Status": "Successful", "Code": 1}
            errorcodelist.append(dict)

    response = simplejson.dumps(errorcodelist[0])

    return response

@route("/api/aaip/python/recommend/board", methods=['POST'])
def get_similar_board_from_item(request):
    print('starting similar board recommendation using board_id')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["clientId"]
        board_id = jsondata["boardId"]
        no_of_boards = jsondata["noOfBoards"]
    except:
        dict = {"errorCode": 701}
        response = simplejson.dumps(dict)
        return response

    users = get_user(client_id)
    errorcodelist = []
    if validate_client_id(client_id):
        dict = {"errorCode": 703}
        errorcodelist.append(dict)


    elif users[0].tolist():
        board_ratings, similar_board, user_ratings_item = itembased_recommendations.load_item_pickle(client_id)
        (start_time,output,end_time) = itembased_recommendations.item_similar_board(similar_board, board_id, no_of_boards)
        print("---item_similar_board recommendations is completed in %s seconds ---" % (end_time - start_time))
        list1 = output
        dict = {"boardId":list1, "status": "successful" , "code":1}
        errorcodelist.append(dict)

    response = simplejson.dumps(errorcodelist[0])

    return response


@route("/api/aaip/python/recommend/item", methods=['POST'])
def get_recommendation_item(request):
    print('starting item_based recommendations')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        user = jsondata["User"]
        no_of_recommendations = jsondata["No of Recommendations"]
    except:
        dict = {"Error_Code": 701}
        response = simplejson.dumps(dict)
        return response

    users = get_user(client_id)
    errorcodelist = []
    if validate_client_id(client_id):
        dict = {"error_code": 703}
        errorcodelist.append(dict)

    elif len(users) > 0 and str(user) not in users[0].tolist():
        dict = {"error_code": 702}
        errorcodelist.append(dict)

    elif str(user) in users[0].tolist():
        board_ratings, similar_board, user_ratings_item = itembased_recommendations.load_item_pickle(client_id)
        (start_time,output,end_time) = itembased_recommendations.generate_recommendation(board_ratings, similar_board, user_ratings_item, user, no_of_recommendations)
        print("---item_based recommendations is completed in %s seconds ---" % (end_time - start_time))
        list1 = str(list(output.index[0:len(output)]))
        dict = {"Board_id":list1, "Status": "Successful" , "Code":1}
        errorcodelist.append(dict)

    response = simplejson.dumps(errorcodelist[0])

    return response


@route("/api/aaip/python/recommend/peer", methods=['POST'])
def get_recommendation_peer(request):
    print('starting peer_based recommendations')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        user = jsondata["User"]
        no_of_recommendations = jsondata["No of Recommendations"]
    except:
        dict = {"Error_Code": 701}
        response = simplejson.dumps(dict)
        return response


    users = get_user(client_id)
    errorcodelist = []
    if validate_client_id(client_id):
        dict = {"error_code": 703}
        errorcodelist.append(dict)

    elif len(users) > 0 and str(user) not in users[0].tolist():
        dict = {"error_code": 702}
        errorcodelist.append(dict)

    elif str(user) in users[0].tolist():
        user_matrix_revised, user_completion_list = peerbased_recommendations.load_peer_pickle(client_id)
        (start_time,output,end_time) = peerbased_recommendations.generate_recommendation(user,user_matrix_revised,user_completion_list,no_of_recommendations)
        print("---peer_based recommendations is completed in %s seconds ---" % (end_time - start_time))
        list1 = str(list(output.index[0:len(output)]))
        dict = {"Board_id": list1, "Status": "Successful", "Code": 1}
        errorcodelist.append(dict)

    response = simplejson.dumps(errorcodelist[0])

    return response


@route("/api/aaip/python/train/user", methods=['POST'])
def get_training_user(request):
    print('starting user_based training')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        board_with_tags = jsondata["BoardWith_Tags"]
        completion_board = jsondata["Completion_Board"]
        endorse = jsondata["Endorse"]
        follow = jsondata["Follow"]
        likes = jsondata["Likes"]
        dummy = jsondata["Dummy"]

    except:
        dict = {"Error_Code": 601}
        response = simplejson.dumps(dict)
        return response



    list_files = [board_with_tags, completion_board, endorse, follow, likes, dummy]

    print ("Checking file format")

    for filename in list_files:
        if file_format_checker(filename):
            dict = {"Error_Code": 602}
            response = simplejson.dumps(dict)
            return response

    print ("File format check is complete")

    print ("Checking File not found")

    for filename in list_files:
        os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
        if not os.path.isfile(filename):
            dict = {"Error_Code": 600}
            response = simplejson.dumps(dict)
            return response
    print ("File not found check complete")

    os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)

    (start_time, user_board_vector, learning_board_vector, completion_series, end_time) = load_user_csv(board_with_tags,
                                                                                                        completion_board,
                                                                                                        endorse, follow,
                                                                                                        likes, dummy)
    os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
    print("saving Pickle Files for user_based training")
    user_board_vector.to_pickle('UserBoardVector.pkl')
    learning_board_vector.to_pickle('LearningBoardVector.pkl')
    completion_series.to_pickle('CompletionSeries.pkl')
    print("---Userbased Training is completed in %s seconds ---" % (end_time - start_time))
    print ('Training Completed, Model saved to Client Model directory, please stop the code execution')

    dict = {"Status": 1}
    response = simplejson.dumps(dict)
    return response


@route("/api/aaip/python/train/board", methods=['POST'])
def get_training_board(request):
    print('Starting item_board training')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["clientId"]
        completion_board = jsondata["completionBoard"]

    except:
        dict = {"errorCode": 601}
        response = simplejson.dumps(dict)
        return response

    os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
    list_files = [completion_board]

    print ("Checking file format")

    for filename in list_files:
        if file_format_checker(filename):
            dict = {"errorCode": 602}
            response = simplejson.dumps(dict)
            return response

    print ("File format check is complete")

    print ("Checking File not found")

    for filename in list_files:
        os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
        if not os.path.isfile(filename):
            dict = {"errorCode": 600}
            response = simplejson.dumps(dict)
            return response

    print ("File not found check complete")

    start_time, user_ratings_item, end_time = load_item_csv(completion_board, client_id)
    print("---board_similarity_based Training is completed in %s seconds ---" % (end_time - start_time))

    print ("Training is completed please wait for pickling to be done")

    os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
    board_ratings, similar_board = similar(user_ratings_item)
    user_ratings_item.to_pickle('UserRatings_Item.pkl')
    board_ratings.to_pickle('BoardRatings.pkl')
    similar_board.to_pickle('similar_board.pkl')
    print ("Pickling Done")
    dict = {"status": 1}
    response = simplejson.dumps(dict)
    return response

@route("/api/aaip/python/train/item", methods=['POST'])
def get_training_item(request):
    print('Starting item_based training')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        completion_board = jsondata["Completion_Board"]

    except:
        dict = {"Error_Code": 601}
        response = simplejson.dumps(dict)
        return response


    os.chdir(WFTP_HOME + '\\Data' + '\\'+ client_id)
    list_files = [completion_board]

    print ("Checking file format")

    for filename in list_files:
        if file_format_checker(filename):
            dict = {"Error_Code": 602}
            response = simplejson.dumps(dict)
            return response

    print ("File format check is complete")


    print ("Checking File not found")

    for filename in  list_files:
        os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
        if not os.path.isfile(filename):
                dict = {"Error_Code": 600}
                response = simplejson.dumps(dict)
                return response

    print ("File not found check complete")


    start_time,user_ratings_item,end_time = load_item_csv(completion_board,client_id)
    print("---Item_based Training is completed in %s seconds ---" % (end_time - start_time))
      
    print ("Training is completed please wait for pickling to be done")
        
    os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
    board_ratings,similar_board = similar(user_ratings_item)
    user_ratings_item.to_pickle('UserRatings_Item.pkl')
    board_ratings.to_pickle('BoardRatings.pkl')
    similar_board.to_pickle('similar_board.pkl') 
    print ("Pickling Done")
    dict = {"Status": 1}
    response = simplejson.dumps(dict)
    return response


@route("/api/aaip/python/train/peer", methods=['POST'])
def get_training_peer(request):
    print('starting peer_based training')
    os.chdir(WFTP_HOME + '\\Data')
    input_params = request.content.getvalue()
    try:
        jsondata = simplejson.loads(input_params)
        client_id = jsondata["ClientID"]
        user_detail = jsondata["User_Detail"]
        completion_board = jsondata["Completion_Board"]
    except:
        dict = {"Error_Code": 601}
        response = simplejson.dumps(dict)
        return response


    os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)


    list_files = [user_detail, completion_board]

    print ("Starting file format check")

    for filename in list_files:
        if file_format_checker(filename):
            dict = {"Error_Code": 602}
            response = simplejson.dumps(dict)
            return response

    print ("File format check is complete")

    print ("Starting checking File not found")

    for filename in list_files:
        os.chdir(WFTP_HOME + '\\Data' + '\\' + client_id)
        if not os.path.isfile(filename):
                dict = {"Error_Code": 600}
                response = simplejson.dumps(dict)
                return response

    print ("File not found check complete")


    print ("Training is completed please wait for pickling to be done")
    
    start_time,user_matrix_revised,user_completion_list,end_time = load_peer_csv(user_detail,completion_board,client_id)

    os.chdir(WFTP_HOME + '\\Model' + '\\' + client_id)
    user_matrix_revised.to_pickle('User_matrix_Revised.pkl')
    user_completion_list.to_pickle('UserCompletionList.pkl')
    print ("Pickling Done")
    print("---peer_based Training is completed in %s seconds ---" % (end_time - start_time))

    dict = {"Status": 1}
    response = simplejson.dumps(dict)
    return response
    

	
run("localhost", 5000)
    

    
