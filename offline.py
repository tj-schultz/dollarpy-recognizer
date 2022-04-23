"""
name: offline.py -- dollargeneral-recognizer
description: alternative launch to perform a recognition loop
and log results offline
authors: TJ Schultz, Skylar McCain, Spencer Bass
date: 4/20/22
"""

from cgi import test
from curses import keyname
from xml.dom.minidom import parse
import xml.dom.minidom as xmlmd
import csv
import os
import path as pth
import nrecognizer as rec
import dollar
import random
import pandas as pd
import time

## list of path types to read using filenames with '01-10' appended
xml_filetypes_unistroke = ["arrow", "caret", "check", "circle", "delete_mark", "left_curly_brace",\
                 "left_sq_bracket", "pigtail", "zig_zag", "rectangle",\
                 "right_curly_brace", "right_sq_bracket", "star", "triangle",\
                 "v", "x"]
xml_filetypes_multistroke_MMG = ["arrowhead", "asterisk", "D", "exclamation_point", "five_point_star",\
                 "H", "half_note", "I", "line", "N", "null", "P", "pitchfork", "six_point_star", \
                 "T", "X"]
xml_filetypes_multistroke_ND = ["arrowhead", "asterisk", "D", "exclamation", "five-point star",\
                 "H", "half-note", "I", "line", "N", "null", "P", "pitchfork", "six-point star", \
                 "T", "X"]

## base dictionary of unprocessed input Path objects from xml
xml_base = {}



## reads a single xml file as a DOM element, records gesture and returns the path object
def read_XML_path(filepath):
  
     ## formed path
    multistrokes = [] 
    xml_path = pth.Path()

    try:
        ## grab root of file
        tree = xmlmd.parse(filepath)
        element = tree.getElementsByTagName("Gesture")[0]

        #parse the stroke tags
        stroke_tags = element.getElementsByTagName("Stroke")
        for stroke in stroke_tags:
             ## parse the point tags
            point_tags = stroke.getElementsByTagName("Point")

            ## get attributes and build path object
            for point in point_tags:
                x = float(point.getAttribute("X"))
                y = float(point.getAttribute("Y"))
                xml_path.stitch(pth.Point(x, y))
            
            multistrokes.append(xml_path.parsed_path)

        # text deebugging to find which files contained incomplete data
        # if len(xml_path) < 2:
        #     print(filepath)
        
    except Exception as e:
        print(e)
        print("Unable to read", filepath, "\n")
        
    return multistrokes

def random100_test(test_files):
    R = rec.Recognizer(useProtractor=False, input_templates=True)

    # dictionary to represent columns in output cvs file - converted to dataframe at end
    output = {
        'User' : [],
        'GestureType' : [],
        'RandomIteration' : [],
        'NumberOfTraningExamples(E)': [],
        'TotalSizeOfTrainingSet' : [],
        'TrainingSetContents' : [],
        'Candidate' : [],
        'RecoResultGesture' : [],
        'Correct/Incorrect' : [],
        'RecoResultScore' : [],
        #'RecoResultBestMatch' : [],
        'RecoResultN-BestList' : []
    }
    # dictionary to store random list of templates each repetion of random100, size = 16 * e
    templates = {}
    # to store one randomly selected canidate for each gesture, size = 16
    canidates = {}

    # score to calculate overall user accuracy
    score = 0
    total=0
    e_scores = [0] * 9
    for user in test_files:
        start = time.time()
        for e in range(1,9):
            for i in range(1,101):
                for gesture in test_files[user]:
                    templates[gesture] = []
                    canidates[gesture] = {}
                    random_temps = random.sample(test_files[user][gesture].keys(), e + 1)
                    canidates[gesture] = random_temps.pop()
                    for temp in random_temps:
                        # add templates to recognizer multistroke set
                        R.add_gesture(gesture,test_files[user][gesture][temp])
                        templates[gesture].append(temp)
                for gesture in canidates.keys():
                        candidate = canidates[gesture]
                        #call recognizer on canidate with list of randomly generated templates from above
                        result = R.recognize(test_files[user][gesture][candidate])

                        n_best = result.n_best
                    
                        #write row elements to output dictionary
                        output['User'].append(user)
                        output['GestureType'].append(gesture)
                        output['RandomIteration'].append(i)
                        output['NumberOfTraningExamples(E)'].append(e)
                        output['TotalSizeOfTrainingSet'].append(e*16)
                        output['TrainingSetContents'].append(list(templates.keys()))
                        output['Candidate'].append(candidate)

                        #gets string of result gesture type
                        reco_gesture = result.name
                        output['RecoResultGesture'].append(reco_gesture)
                        output['Correct/Incorrect'].append('correct' if reco_gesture == gesture else 'incorrect')
                        if(reco_gesture == "Null" or reco_gesture == "No match."):
                             output['RecoResultScore'].append("empty file")
                        else: 
                            output['RecoResultScore'].append(n_best[result.name])
                            total+=1
                        #olist indices must be integers or slices, not strmc vutput['RecoResultBestMatch'].append(n_best[0][0])
                        output['RecoResultN-BestList'].append(n_best)

                        if(reco_gesture == gesture):
                            #add overall accuracy score
                            score=score+1
                             #compute average accuracy score for each level of e
                            e_scores[e-1]+=1

                templates.clear()
                canidates.clear()
                R.delete_all_templates()  
        end = time.time()  
        print("user ", user, " completed random100 loop")
        print("\t execution time of user ", user, ": ", end-start)
    score_df = pd.DataFrame({'User' : ['AvgUserAccuracy'], 'GestureType' : [score/total], 'RandomIteration' : [''], 'NumberOfTrainingExamples(E)' : [''], 'TotalSizeOfTrainingSet' : [''], 'TrainingSetContents' : [''], 'Candidate' : [''], 'RecoResultGesture' : [''], 'Correct/Incorrect' : [''], 'RecoResultScore' : [''], 'RecoResultBestMatch' : [''], 'RecoResultN-BestList' : ['']})  
    output_df = pd.DataFrame(output)
    output_df = pd.concat([output_df, score_df])
    output_df.to_csv('random100_test_output.csv')

     #output average accuracy for each level of e
    e_avg_err= {'Eaverages': []}
    for lev in range(0,9):
          # e_score/(number of gestures * number of users * number of repetions)
        e_avg_err['Eaverages'].append(1-(e_scores[lev]/16*9*len(test_files.keys())))
   
    e_avg_output = pd.DataFrame(e_avg_err)
    e_avg_output.to_csv('average_for_e_levels.csv')
    
   
if __name__ == "__main__":


    ## build xml_base
    # ## for Multistroke gesture log from N$ site (MMG)
    # for user_key in ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]:   ## for each user
    #     xml_base[user_key] = {}                 ## add user key-dict
    #     for prefix in xml_filetypes_multistroke_MMG:## for each gesture
    #         xml_base[user_key][prefix] = {}     ## add prefix key-dict
    #         for num in range(1, 11):            ## for each sample xml
    #             file_key = str(num).zfill(2)
    #             speed_key = "".join([user_key, str("finger-SLOW")])

    #             ## read as DOM -- append to dictionary
    #             xml_base[user_key][prefix][file_key] = read_XML_path(\
    #                 os.path.join(os.getcwd(), "Samples", user_key, speed_key, "%s-%s.xml"\
    #                              % (prefix, file_key))
    #             )

    ## for New Data
    for user_key in ["S01", "S02", "S03", "S04", "S05", "S06"]:   ## for each user
        xml_base[user_key] = {}                 ## add user key-dict
        for prefix in xml_filetypes_multistroke_ND:## for each gesture
            xml_base[user_key][prefix] = {}     ## add prefix key-dict
            for num in range(1, 11):            ## for each sample xml
                file_key = str(num).zfill(2)

                ## read as DOM -- append to dictionary
                xml_base[user_key][prefix][file_key] = read_XML_path(\
                    os.path.join(os.getcwd(), "NewData", user_key, "%s%s.xml"\
                                 % (prefix, file_key))
                )
    
    print("xml_base finished build")



    random100_test(xml_base)
   
  
    

    ## debug -- vectors should be of length 2 * 64 = 128
    # for user in R.preprocessed:
    #     for gesture in R.preprocessed[user]:
    #         for id in R.preprocessed[user][gesture].keys():
    #             print(user, gesture, id, "length:", len(xml_base[user][gesture][id]))
    #             #R.recognize(R.preprocessed[user][gesture][id], R.preprocessed[user][gesture], preprocess=False)