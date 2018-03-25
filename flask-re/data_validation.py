import os
import glob
import pandas as pd



WFTP_HOME = os.path.abspath(os.environ['WFTP_HOME'])

def empty_cell_checker(dataframe):
    response = ""
    list = dataframe.columns[dataframe.isnull().any()].tolist() 
    if len(list) > 0 :
        response = "Nan values is present in " + str(list)
    else:
        print ("NAN values are not present" ) 
    return response

def validate_client_id(client_id):
    list = os.listdir(WFTP_HOME + '\\Data')

    if client_id not in list:
        return True
    else:
        return False

def item_file_checker(client_id, filename):
    filepath = Path(WFTP_HOME + '\\Data' + '\\' + client_id + "\\" + filename)
    if os.path.exists(filepath):
        return True

def file_format_checker(fname):
    if not (fname.endswith('csv')):
        return True


def validate_csv(client_id):
    path = os.chdir(WFTP_HOME + '\\Data' +'\\'+ client_id)
    allfiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    i = 0
    for file_ in allfiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        dataframe = pd.DataFrame(index=[i+1], columns=['file_name','errors'], dtype=str)
        dataframe['file_name'] = file_
        dataframe['errors'] = empty_cell_checker(dataframe)
        list_.append(dataframe)
        i+= 1
      
    frame = pd.concat(list_)   
    return frame

    

