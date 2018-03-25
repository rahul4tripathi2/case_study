#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy

'''
user_id = [ACC11079227,SHAHC,11079227]
system_id = [S7H,K4A,K4X]
system = [ECC,SRM,CRM]
Client = [050,100,200]
Name = [Chintan, Rahul]
Case_Number = [07466117,INC00005]
Job_Name = [AP_AU_TOLAS_INBOUND,KWIC116UPG,EBIINN15AFT,RPINS033,RPINS102]

'''

# training data
TRAIN_DATA = [

    # user_id training
    ('user_id is ACC11079227', {
        'entities': [(11, 22, 'USER_ID')]
    }),
    ('user_id is SHAHC', {
        'entities': [(11, 16, 'USER_ID')]
    }),
    ('user_id is 10935947', {
        'entities': [(11, 19, 'USER_ID')]
    }),

    ('create new_id', {
        'entities': [(7, 13, 'USER_ID')]
    }),

    # operation training

     ('please reset password', {
        'entities': [(7, 12, 'OPERATION')]
    }),

    ('reset password', {
            'entities': [(0, 4, 'OPERATION')]
        }),

    # system_id training
    ('system_id is S7H', {
        'entities': [(13, 16, 'SYSTEM_ID')]
    }),
    ('system_id is K4A', {
        'entities': [(13, 16, 'SYSTEM_ID')]
    }),
    ('system_id is K4X', {
        'entities': [(13, 16, 'SYSTEM_ID')]
    }),

    ('system_id is S7H', {
            'entities': [(13, 16, 'SYSTEM_ID')]
        }),
        ('system_id is K4A', {
            'entities': [(13, 16, 'SYSTEM_ID')]
        }),
        ('system_id is K4X', {
            'entities': [(13, 16, 'SYSTEM_ID')]
        }),
    
    # system training
    ('system_id is ECC', {
        'entities': [(13, 16, 'SYSTEM')]
    }),

    ('system_id is SRM', {
            'entities': [(13, 16, 'SYSTEM')]
        }),

    ('system_id is CRM', {
            'entities': [(13, 16, 'SYSTEM')]
        }),

    # client_id training
    ('client_id is 050', {
        'entities': [(13, 16, 'CLIENT_ID')]
    }),

   # name_training
    ('Name is Rahul', {
        'entities': [(8, 13, 'NAME')]
    }),

   ('Name is Alex', {
        'entities': [(8, 12, 'NAME')]
    }),

   # case_no training
    ('case_no is 07466117', {
        'entities': [(11, 19, 'CASE_NO')]
    }),
    ('case_no is INC00005', {
        'entities': [(11, 19, 'CASE_NO')]
    }),

   # job_name training

    ('job_name is AP_AU_TOLAS_INBOUND', {
        'entities': [(12, 31, 'JOB_NAME')]
    }),
    ('job_name is KWIC116UPG', {
        'entities': [(12, 22, 'JOB_NAME')]
    }),

    ('job_name is RPINS033', {
        'entities': [(12, 20, 'JOB_NAME')]
    })

]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))


def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""

    print (output_dir)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)


    # test the trained model

    doc = nlp(u'My Name is RAM. user_id 10456789, system_id is X2M, system is GGG, client is 090, case_no is INC55555 ,job_name is RPINS110')

    for ent in doc.ents:
        print(ent.text, ent.label_)

    nlp.to_disk('C:\\Users\\rahul.a.tripathi\\PycharmProjects\\machine-learning\self-heal\\en_model')
    print ("Training is completed please load your model to test your trained model")
    

if __name__ == '__main__':
    plac.call(main)

