
# Set up spaCy and loading library

import spacy
nlp = spacy.load('en')

import pandas as pd
import glob, os
import json
import time


absolute_path = os.path.abspath(os.path.dirname('self_heal_ticket_resolution_v2.py'))
os.chdir(absolute_path)


def similar_resolution(problem_desc):
    start_time = time.time()
    file_name = glob.glob("*.csv")[0]
    train_data = pd.read_csv(file_name)
    train_data = train_data[['Problem Description', 'Solution Description' ]]
    train_data = train_data.replace({r'\r\n': ''}, regex=True)
    result_df = pd.DataFrame()
    nlp_test_data = nlp(unicode(problem_desc))

    for index, data in train_data.iterrows():
        nlp_train_data = nlp(unicode(data[['Problem Description']]))
        sim_score = nlp_train_data.similarity(nlp_test_data)

        if sim_score > 0.0:
            result_df = result_df.append({'resolution': data['Solution Description'], 'score': sim_score}, ignore_index=True)


    result = result_df.sort_values(['score'], ascending=[0]).head(3).to_json(orient='records')
    result = json.loads(result)
    print("--- execution time is %s seconds ---" % (time.time() - start_time))
    return result














