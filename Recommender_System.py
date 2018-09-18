import pandas as pd
import json

def get_training_data(file_name):
    try:
        with open(file_name) as json_file:
            data = {'data': [json.loads(line) for line in json_file]}
        return data
    except:
        print('Please try another file name.')
        return None

# training_data = get_training_data('reviews.training.json')
shortened_training_data = get_training_data('shortened_training_data.json')


# with open('shortened_training_data.json', 'w') as outfile:
#     json.dump(training_data['data'][0:100], outfile)

print(shortened_training_data['data'])
