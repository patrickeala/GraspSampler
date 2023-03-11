import pickle
import os
import json


db = pickle.load(open('models_sampled.pkl', 'rb'), encoding='latin1')

directory = 'training_data_2'

for id in db.keys():
    print(db[id])
    data = db[id]
    cat = db[id]["category"]    
    # if cat == "bottle":
    #     data["scale"] = data["scale"] * 0.9
    path = f"{directory}/{cat}"
    if not os.path.exists(path):
        os.makedirs(path)
    counter = 0
    for source_file_name in os.listdir(f'{path}'):
        counter += 1
    filename = f"{path}/{cat}{counter:03}.json"

    with open(filename, 'w') as fp:
        json.dump(data, fp)
