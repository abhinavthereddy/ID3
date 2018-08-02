import math
import pandas as pd
import numpy as np
import csv
from pandas import DataFrame, read_csv
import pandas as pd
from numpy.linalg import inv


class ID3:
    def __init__ (self):
        self.root = Node(None)
    
    def uncertainty (outcomes):
        outcomes2 = [x / (float(sum(outcomes))) for x in outcomes]
    
        uncertainty = 0.0
        for probability in outcomes2:
            if (probability != 0):
                uncertain = -(probability)*math.log2(probability)
            else:
                uncertain = 0
            uncertainty += uncertain
        return uncertainty

    def information_gain (types_count):

        """ Types: The number of types the parameter has.
            types_count: The number of instances in the each type.

            Eg. Parameter = Outlook
                Types = ['Sunny', 'Rain', 'Overcast']
                types_count = [[2, 3], [3, 2], [4, 0]] (This example has two outcomes, Yes or No)"""

        outcomes_parent = []
        for i in range(len(types_count)):
            for j in range(len(types_count[i])):
                if len(outcomes_parent) < j+1:
                    outcomes_parent.append(types_count[i][j])
                else:
                    outcomes_parent[j] += types_count[i][j]                    

        uncertainty_parent = ID3.uncertainty(outcomes_parent)
        uncertainty_child = []
        for i in types_count:
            uncertainty_child.append(ID3.uncertainty(i))        

        total = sum(outcomes_parent)
        gain = uncertainty_parent
        for i in range(len(types_count)):
            gain -= (float(sum(types_count[i])) * uncertainty_child[i]) / total

        return gain

    def find_best_attribute (df):
        attributes = list(df) [:-1]  #List of all possible attributes. Eg. Outlook, Temperature, Humidity, Wind etc.
        values = list(df)[-1]  # Header name of output
        result = (df[values].tolist())
        possible_results = []  #List with all possible classes. Eg. Yes or No to event of playing tennis.
        for i in result:
            if i not in possible_results:
                possible_results.append(i)

        possible_values_of_attributes = {} #List of list of possible values of each attribute.
        for attribute in attributes:
            possible_values_of_attributes[attribute] = []
            for value in df[attribute].tolist():
                if value not in possible_values_of_attributes[attribute]:
                    possible_values_of_attributes[attribute].append(value)

        dictionary = {}
        for attribute in attributes:
            dictionary[attribute] = {}
            for value in possible_values_of_attributes[attribute]:
                dictionary[attribute][value] = {}
                for result in possible_results:
                    dictionary[attribute][value][result] = 0

        for index, row in df.iterrows():
            outcome = row[values]
            for attribute in attributes:
                dictionary[attribute][row[attribute]][outcome] += 1

        gains = {}
        for attribute in attributes:
            out_of_value = []
            for value in possible_values_of_attributes[attribute]:
                temp = []
                for result in possible_results:
                    temp.append(dictionary[attribute][value][result])
                out_of_value.append(temp)
            gain = ID3.information_gain(out_of_value)
            gains[attribute] = gain


        max_attribute = max(gains, key=gains.get)    
        return max_attribute
    
    def create_child(parent_node, df):
        if (len(df.columns) < 2):
            print("Invalid Data Frame")


        attributes = list(df) [:-1]
        possible_values_of_attributes = {} #List of list of possible values of each attribute.
        for attribute in attributes:
            possible_values_of_attributes[attribute] = []
            for value in df[attribute].tolist():
                if value not in possible_values_of_attributes[attribute]:
                    possible_values_of_attributes[attribute].append(value)

        values = list(df)[-1]  # Header name of output

        best_attribute = ID3.find_best_attribute(df)

        Node.set_attribute(parent_node, best_attribute)

        attribute_children_dict = {}   

        for attribute in possible_values_of_attributes[best_attribute]:        
            outcomes = df[df[best_attribute] == attribute][values]
            outcomes = outcomes.unique()
            if len(outcomes) == 1:                
                attribute_children_dict[attribute] = Node(parent_node, attribute = None, attribute_children_dict = None, leaf_node = True, outcome = outcomes[0])            
            else:                
                df_for_child = df[df[best_attribute] == attribute].drop(columns=best_attribute)
                attribute_children_dict[attribute] = Node(parent_node)
                ID3.create_child(attribute_children_dict[attribute], df_for_child)            

        Node.set_attribute_children_dict(parent_node, attribute_children_dict)


    def create_tree(self, df):      
        ID3.create_child(self.root, df)    

    def predict_outcome(self, df):
        for index, row in df.iterrows():
            values = list(df)[-1]
            pointer = self.root        

            while(not Node.get_leaf_node(pointer)):
                attribute = Node.get_attribute(pointer)
                pointer = Node.get_attribute_children_dict(pointer)[row[attribute]]            
            outcome = Node.get_outcome(pointer)
#             row[values] = outcome
            df.loc[index, values] = outcome
        return df
            
          

class Node(object):
    
    def __init__(self, parent, attribute = None, attribute_children_dict = None, leaf_node = None, outcome = None):
        self.parent = parent
        self.attribute = attribute
        self.attribute_children_dict = attribute_children_dict
        self.leaf_node = leaf_node
        self.outcome = outcome
        
    def get_parent(self):
        return self.parent
    
    def get_attribute_children_dict(self):
        return self.attribute_children_dict
    
    def get_attribute(self):
        return self.attribute
    
    def set_attribute(self, attribute):
        self.attribute = attribute
    
    def set_attribute_children_dict(self, attribute_children_dict):
        self.attribute_children_dict = attribute_children_dict
        
    def get_leaf_node(self):
        return self.leaf_node
               
    def set_leaf_node(self, yes_or_no):
        self.leaf_node = True
        self.yes_or_no = yes_or_no
        
    def get_outcome(self):
        if self.leaf_node:
            return self.outcome
        

