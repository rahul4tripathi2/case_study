# Importing library
import pandas as pd
import spacy
nlp = spacy.load('C:\\Users\\rahul.a.tripathi\\PycharmProjects\\machine-learning\\self-heal\\en_model')

'''
# self heal entities
user_id = [ACC11079227,SHAHC,11079227]
system_id = [S7H,K4A,K4X]
system = [ECC,SRM,CRM]
Client = [050,100,200]
Name = [Chintan, Rahul]
Case_Number = [07466117,INC00005]
Job_Name = [AP_AU_TOLAS_INBOUND,KWIC116UPG,EBIINN15AFT,RPINS033,RPINS102]
'''

'''
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

'''

'''
data = pd.read_csv('data.csv')
data = data[['Incident ID', 'Problem Description']]
data = data.replace({r'\r\n': ''}, regex=True)
doc = nlp(unicode(data[['Problem Description']]))
'''

data = nlp(u'My Name is Mohan, user id is 10945676, system_id is X4J, system is JJJ, client is 090, case_no is INC11122 , job name is AP_AU_TOLAS_OUTBOUND')

# extracting all entities
name_entity = [ent for ent in data.ents if ent.label_ == 'NAME']
print "name_entity are : " + str(name_entity)

user_id_entity = [ent for ent in data.ents if ent.label_ == 'USER_ID']
print "user_id_entity are : " + str(user_id_entity)

system_id_entity = [ent for ent in data.ents if ent.label_ == 'SYSTEM_ID']
print "system_id_entity are : " + str(system_id_entity)

system_id_entity = [ent for ent in data.ents if ent.label_ == 'SYSTEM']
print "system_id_entity are : " + str(system_id_entity)

client_id_entity = [ent for ent in data.ents if ent.label_ == 'CLIENT_ID']
print "client_id_entity are : " + str(client_id_entity)

case_no_entity = [ent for ent in data.ents if ent.label_ == 'CASE_NO']
print "case_no_entity are : " + str(case_no_entity)

job_name_entity = [ent for ent in data.ents if ent.label_ == 'JOB_NAME']
print "job_name_entity are : " + str(job_name_entity)

operation_entity = [ent for ent in data.ents if ent.label_ == 'OPERATION']
print "operation_entity are : " + str(operation_entity)