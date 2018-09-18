import pandas as pd
import json

# Returns training data as a Pandas dataframe.
def get_training_data(file_name):

    try:
        with open(file_name) as json_file:
            data = {'data': json.loads(line) for line in json_file}
        return pd.DataFrame(data['data'], columns=['reviewerID', 'asin', 'reviewerName', 'helpful',
                             'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime'])
    except:
        print('Please try another file name.')
        return None

shortened_training_data = get_training_data('shortened_training_data.json')

print(shortened_training_data.head())