import json
import pandas as pd
import boto3
import os
import scipy.stats as st
import numpy as np
from IPython.display import display, HTML

PATH = "data/fairface/fairface_images/"
SAVING_ITER = 1000

reko = boto3.client("rekognition")

# Creates a dict of bytes from an image which can 
# then be fed into Rekognition
def create_image_dict_from_file(photo):
    with open(photo, 'rb') as image:
        return {'Bytes': image.read()}
    
# Turns JSON encoded data into Python object
def load_from_json(file):
    with open(file, 'r', encoding='utf-8') as fp:
        return json.load(fp)
    
# Saves data to json file
def save_to_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

# Combines data/fairface/fairface_label_val.csv and 
# data/fairface/fairface_label_train.csv into one
# dataframe
def labels_to_df():
    df = pd.read_csv("data/fairface/fairface_label_val.csv").append(pd.read_csv("data/fairface/fairface_label_train.csv"))
    df.rename(columns={"gender": "Gender"}, inplace=True)
    df.rename(columns={"race": "Race"}, inplace=True)
    df.rename(columns={"file": "Filename"}, inplace=True)
    df.sort_values(by=["Filename"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

# Given a JSON file, returns the file transformed 
# into a pandas dataframe
def results_to_df():
    results = load_from_json("newFairFace.json")
    df = pd.DataFrame.from_dict(results, "index")
    df.index.name = "Filename"
    df.sort_values(by=["Filename"], inplace=True)
    df.reset_index(inplace=True)
    return df

# Returns two dataframes, one with labels
# where rekognition returned results,
# and the other where it did not return results
def fairface_results_intersection(labels, results):
    bool_labels_with_results = labels["Filename"].isin(results["Filename"])
    return labels[bool_labels_with_results], labels[~bool_labels_with_results]

# Creates the results and labels dataframes, returning
# a dataframe of the labels and results as well as a 
# dataframe of the pictures that did not return results
def combine_labels_and_results():
    results_df = results_to_df()
    labels_df = labels_to_df()
    labels_with_results, labels_without_results = fairface_results_intersection(labels_df, results_df)
    complete_df = pd.DataFrame(data=list(results_df["Filename"]), columns=["Filename"])
    complete_df = complete_df.assign(Race = list(labels_with_results["Race"]),
                                    Label = list(labels_with_results["Gender"]),
                                    Prediction = [x["Gender"]["Value"] for x in results_df["details"]],
                                    Confidence = [x["Gender"]["Confidence"] for x in results_df["details"]],
                                    Faces = list(results_df["num_faces"]),
                                    Boundingbox = [x["BoundingBox"] for x in results_df["details"]])
    return complete_df, labels_without_results

# Computes Wilson Confidence Interval and returns upper
# and lower bounds
def wilson_ci(num_hits, num_total, confidence=0.95):
    z = st.norm.ppf((1+confidence)/2)
    phat = num_hits / num_total
    first_part = 2*num_total*phat + z*z
    second_part = z*np.sqrt(z*z - 1/num_total + 4*num_total*phat*(1-phat) + (4*phat-2))+1
    den = 2*(num_total + z*z)
    lower_bound = max(0,(first_part - second_part) / den) if phat!=0 else 0
    upper_bound = min(1,(first_part + second_part) / den) if phat!=1 else 1
    return lower_bound,upper_bound

# Calculates metrics for given labels and predictions at 
# given confidence threshold, returning a dict of the
# results
def metrics_calculator(label_list, pred_list, conf_list, conf_threshold):
    label_set = sorted(set(label_list))
    gender_dict = dict()

    for gender in label_set:
        tp = 0
        fp = 0
        num_pos = 0
        num_neg = 0
        for i,label in enumerate(label_list):
            pred = pred_list[i]
            conf = conf_list[i]
            bool_conf = (conf >= conf_threshold)
            bool_gender_match = (pred==gender and label==gender)
            bool_gender_mismatch = (pred==gender and label!=gender)
            tp = tp + (bool_conf and bool_gender_match)
            fp = fp + (bool_conf and bool_gender_mismatch)
            num_pos = num_pos + (label == gender)
            num_neg = num_neg + (label != gender)
        
        metric_dict = dict()
    
        metric_dict["Precision"] = tp / (tp + fp)
        metric_dict['Prec_ci_lower'], metric_dict['Prec_ci_upper'] = wilson_ci(tp, tp+fp)
    
        metric_dict["FDR"] = 1 - metric_dict["Precision"]
        metric_dict['FDR_ci_lower'], metric_dict['FDR_ci_upper'] = wilson_ci(fp, tp+fp)
    
        metric_dict["Recall"] = tp / num_pos
        metric_dict['Reca_ci_lower'], metric_dict['Reca_ci_upper'] = wilson_ci(tp, num_pos)
    
        metric_dict["FPR"] = fp / num_neg
        metric_dict['FPR_ci_lower'], metric_dict['FPR_ci_upper'] = wilson_ci(fp, num_neg)
    
        metric_dict["FNR"] = 1 - metric_dict["Recall"]
        metric_dict['FNR_ci_lower'], metric_dict['FNR_ci_upper'] = wilson_ci(num_pos-tp, num_pos)
    
        metric_dict["TP"] = tp
        metric_dict["FP"] = fp
        metric_dict["TP+FP"] = tp + fp
    
        metric_dict["num_pos"] = num_pos
        metric_dict["num_neg"] = num_neg
    
        gender_dict[gender] = metric_dict
    
    return gender_dict

# Returns a dataframe of metrics for all races
def metrics_dataframe(df, conf_threshold):
    result_dict = dict()
    race_list = list(df["Race"])
    
    result_dict["All"] = metrics_calculator(list(df["Label"]), list(df["Prediction"]), 
                                            list(df["Confidence"]), conf_threshold)
    for race in set(race_list):
        race_df = df[df["Race"] == race]
        label_list = list(race_df["Label"])
        pred_list = list(race_df["Prediction"])
        conf_list = list(race_df["Confidence"])
        
        race_dict = metrics_calculator(label_list, pred_list, conf_list, conf_threshold)
        result_dict[race] = race_dict

    return pd.DataFrame.from_dict(result_dict, orient="index"), result_dict

# Returns a dataframe of the number of males and females
# in each subgroup
def groupsize_to_df(df, races):
    size_dict = dict()
    for race in races:
        rdf = df[df["Race"] == race]
        size_dict[race] = [len(rdf[rdf["Label"] == "Female"]), len(rdf[rdf["Label"] == "Male"])]
        rdf = pd.DataFrame.from_dict(size_dict)
        
        d = dict(selector="th", props=[('text-align', 'center')])
        
        retdf = rdf.rename(index={0: "Female", 1: "Male"})
        retdf = retdf.style.set_properties(**{"width":"10em", "text-align":"center"}).set_table_styles([d])
    return retdf

# Runs FairFace photos through Rekogntion, saves the 
# results to newFairFace.json, and returns the dict
def create_details():
    try:
        all_details = load_from_json("newFairFace.json")
    except FileNotFoundError:
        all_details = dict()
    
    labels_df = labels_to_df()
    filename_list = list(labels_df["Filename"])

    for i,file in enumerate(filename_list):
        face_details = dict()
    
        if file in all_details:
            continue
    
        image_bytes = create_image_dict_from_file(PATH + file)
        response = reko.detect_faces(Image=image_bytes, Attributes=["ALL"])
        num_faces = len(response["FaceDetails"])
    
        if num_faces == 0:
            continue
        
        face_details["details"] = response["FaceDetails"][0]
        face_details["num_faces"] = num_faces
        
        all_details[file] = face_details
        
        if i % SAVING_ITER == 0:
            print("Doing backup save")
            save_to_json(all_details, "newFairFace.json")
    
    save_to_json(all_details, "newFairFace.json")
    return all_details


    