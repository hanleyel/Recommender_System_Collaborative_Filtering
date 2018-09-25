import pandas as pd
import json
import numpy as np
from scipy.sparse import csr_matrix


############################################
#####     LOADING AND STORING DATA     #####
############################################

# Output json training data as a Pandas dataframe.
def get_training_data(file_name):

    try:
        with open(file_name) as json_file:
            data = {'data': json.loads(line) for line in json_file}
            data_frame = pd.DataFrame(data['data'], columns=['reviewerID', 'asin', 'reviewerName', 'helpful',
                             'reviewText', 'overall', 'summary', 'unixReviewTime', 'reviewTime'])
            return data_frame
    except:
        print('Please try another file name.')
        return None


# Convert Pandas dataframe to csv file for storage purposes.
# NOTE: Don't run this with the actual training data. This was just for saving a small version of the file for time
# saving purposes while I was setting up my dataframe and matrices.
def convert_to_csv(dataframe, desired_filename):

    try:
        return dataframe.to_csv(desired_filename, index=False)
    except:
        print('Please try another dataframe or file name.')


########################################################
#####     CONVERTING DATAFRAME TO A CSR MATRIX     #####
########################################################

# Create Reviewer x Product matrix populated with ratings
def create_reviewer_product_matrix(dataframe):

    # Pivot the dataframe so that unique reviewers are on the y axis and unique products are on the x axis.
    pivot_dataframe = dataframe.pivot(index='reviewerID', columns='asin', values='overall')

    # Convert the dataframe to a matrix.
    # This matrix still contains NaN values.
    reviewer_product = csr_matrix(pivot_dataframe.values)

    return reviewer_product


#####################################
#####     RUNNING FUNCTIONS     #####
#####################################

training_data = get_training_data('shortened_training_data.json')

# convert_to_csv(shortened_training_data, 'shortened_training_data.csv')

reviewer_product_matrix = create_reviewer_product_matrix(training_data)

####################################
#####     PRINT STATEMENTS     #####
####################################

# print(shortened_training_data[['reviewerID', 'asin', 'reviewerName', 'helpful', 'overall', 'summary']].head())
# print(shortened_training_data['reviewerID'].value_counts())
# print(shortened_training_data['asin'].value_counts())
print(reviewer_product_matrix)