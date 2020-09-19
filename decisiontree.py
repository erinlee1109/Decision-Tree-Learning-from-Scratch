"""
Yujeong (Erin) Lee
CS 365 Lab C
decisiontree.py
"""

import math
import csv
import argparse
import os
import copy

def readfile(inputfile):
    with open(inputfile, newline = '') as data:                                                                                          
        data_reader = csv.reader(data, delimiter='\t')
        training_data = []
        headers = next(data_reader, None)
        for data in data_reader:
            training_data.append(data)
    return(headers, training_data)

def unique_option(input_data, col):
    result = []
    for row in input_data:
        if row[col] not in result:
            result.append(row[col])
    return(result)


"""E N T R O P Y""" 
"""To build a decision tree, we need to first calculate two types of entropy using frequency. 
Entropy using the frequency of one attribute"""
def entropy_one_attr(input_data, col):
    yes = 0
    no = 0 
    for row in input_data: 
        decision = row[-1]
        if decision == 'yes':
            yes+=1
        else:
            no+=1
    count = (yes, no)
    total = int(yes)+int(no)

    """calculate the probability"""
    prob = []
    for i in count: 
        prob.append(i/total) 

    """calculate entropy for the last column"""
    if prob[0] == 1 or prob[1] == 1: 
        entropy = 0
    elif prob[0] == 0.5 or prob[1] == 0.5:
        entropy = 1
    else: 
        entropy = -(prob[0]*math.log2(prob[0])) - (prob[1]*math.log2(prob[1]))

    return (entropy)



"""Entropy using the frequency of two attributes. 
Entropy for the selected column and the last column (desicion column w/ yes and no)"""
def entropy_two_attr(input_data, col):
    final_entropy = 0
    decision_total = 0
    for row in input_data: 
        decision_total += 1
    for option in unique_option(input_data, col):
        """for each attribute option, calculate the number of yes and no"""
        yes = 0
        no = 0 
        for row in input_data: 
            decision = row[-1]
            if row[col] == option and decision == 'yes':
                yes+=1
            elif row[col] == option and decision == 'no':
                no+=1
                
        """establishing variables"""
        count = (yes, no)
        attr_total = int(yes)+int(no)
        prob = []
        
        """calculate the probability"""
        for i in count: 
            prob.append(i/attr_total) 
        
        """for each attribute option, calculate entropy"""
        if prob[0] == 1 or prob[1] == 1: 
            entropy = 0
        elif prob[0] == 0.5 or prob[1] == 0.5:
            entropy = 1
        else: 
            entropy = -(prob[0]*math.log2(prob[0])) - (prob[1]*math.log2(prob[1]))
            
        final_entropy += (attr_total/decision_total)*float(entropy)
    return final_entropy


def info_gain(input_data, col):
    return(entropy_one_attr(input_data, col) - entropy_two_attr(input_data, col))


def attr_w_highest_info_gain(input_data):
    col_count = (len(input_data[0])) 
    highest_info_gain = 0
    chosen_attr_index = ''
    for i in range(col_count-1): # excluding the last column that contain yes/no
        if info_gain(input_data, i) > highest_info_gain:  
            highest_info_gain = info_gain(input_data, i)
            chosen_attr_index = i
    return(highest_info_gain, chosen_attr_index)


def split_by_chosen_attr(input_data, chosen_attr_index): 
    """create empty lists according to number of unique options in the chosen attribute"""
    n = (len(unique_option(input_data, chosen_attr_index)))
    lists = [[] for i in range(n)]
    
    """split by options into separate lists"""
    unique = (unique_option(input_data, chosen_attr_index))
    for row in input_data: 
        for index in range(len(unique)): 
            if row[chosen_attr_index] == unique[index]:
                lists[index].append(row)
    return lists


class Node:
    def __init__(self, option, attribute, children=[]):
        self.option = option # unique options of previously chosen attribute
        self.attribute = attribute # next attribute according to which children are determined
        self.children = children

        
class Leaf: 
    def __init__(self, prediction):
        self.prediction = prediction

        
def build_tree(root, input_data):
    """However many lists are returned in the lists of lists, recursively build tree on each option here."""
    gain, chosen_attr_index = attr_w_highest_info_gain(input_data)
    
    if (gain == 0) or (chosen_attr_index == ''): # base case
        prediction = input_data[0][-1]
        child = Leaf(prediction)
        root.children.append(child)
        return root
    
    else: 
        root.attribute = headers[chosen_attr_index]        
        current_data = split_by_chosen_attr(input_data, chosen_attr_index)
        children = []

        for i in range(len(current_data)): # 0 1 2 
            unique = unique_option(current_data[i], chosen_attr_index)
            child = Node(unique, None, [])            
            child = build_tree(child, current_data[i])
            children.append(child)

        root.children = children
        return root

    
def print_tree(root, indent=""):
    if isinstance(root, Leaf):
        print(indent + root.prediction)
        return
     
    print(indent + str(root.option) + ": " + str(root.attribute))
    
    for child in root.children: 
        print_tree(child, indent + "| ")
        

def classify(rowDict, node):
    if isinstance(node.children[0],Leaf):
        return node.children[0].prediction
    
    option = rowDict[node.attribute]
    child_index = 0
    try:
        while(node.children[child_index].option != [option]):
            child_index += 1
    except IndexError:
        raise ValueError("UnknownValueError")
        
    child=node.children[child_index]
    return classify(rowDict,child)


def accuracy_test(input_data):
    """Leave-one-out cross-validation accuracy test"""
    count = -1
    copy1 = copy.deepcopy(input_data)
    dicts = [] 
    for row in copy1: 
        dict = {}
        for i in range(len(headers)-1):
            dict.update({headers[i]:row[i]})
        count += 1
        dict.update({"index" : count})
        dict.update({"expected" : row[-1]})
        dicts.append(dict)
    total = len(dicts)
    
    count2 = 0
    for i in dicts: 
        copy2 = copy.deepcopy(input_data)
        copy2.pop(i['index'])
        root = Node(None, None, [])
        root = build_tree(root, copy2)
        try: 
            decision = classify(i, root)
        except ValueError:
            print("value error at ", i)
            pass
        if decision == i["expected"]:
            count2 += 1 
    
    # print(str(count2) + "/" + str(total))
    return("Accuracy: " + str(count2/total*100)+"%")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Lab C - Classifier")
    parser.add_argument('-i','--inputFileName', type=str, help='Rows of data', required=True)
    args = parser.parse_args()
    
    if not (os.path.isfile(args.inputFileName)):
        print("error,", args.inputFileName, "does not exist, exiting.", file=sys.stderr)
        exit(-1)
    
    headers, training_data = readfile(args.inputFileName)
    root = Node(None, None, [])
    build_tree(root, training_data)
    print_tree(root)
    print(accuracy_test(training_data)) 
    
    exit(0)
    