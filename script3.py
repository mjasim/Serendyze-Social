import csv 
import json

# this script is being run for only one csv file, in this case gunControl.csv
# small modifications can be made to make it suitable for two objects
def csv_to_json(csv_file_path, json_file_path):
    # innermost dictionary
    data_dict = {} 
    # second list 
    tempList = []
    # penultimate dictionary
    outerDict = {}
    # outermost list
    finalList = []
    # reads the file as a csv
    with open(csv_file_path, encoding = 'utf-8') as csv_file_handler:
        # reads the dictionary
        csv_reader = csv.DictReader(csv_file_handler)
        # iterates through the csv file row by row 
        for rows in csv_reader:
            # creates an object of this list 
            tempList.append(rows)
        # creates a dictionary with key value pairs to input into the penultimate dictionary
        outerDict["comments"] = tempList
        # appending to the final list
        finalList.append(outerDict)


    with open(json_file_path, 'w', encoding='utf-8') as json_file_handler:
        json_file_handler.write(json.dumps(finalList, indent = 4))

csv_file_path = input("Enter csv path name")
json_file_path = input("Enter json file path")
csv_to_json(csv_file_path, json_file_path)
