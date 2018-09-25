import pandas as pd
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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

# Pivot datarame and create Reviewer x Product matrix populated with ratings. Return tuple with sparse matrix and dataframe.
def create_reviewer_product_matrix(dataframe):

    # Pivot the dataframe so that unique reviewers are on the y axis and unique products are on the x axis.
    # NOTE: Removed zeros in order to perform algebraic operations.
    reviewer_product_dataframe = dataframe.pivot(index='reviewerID', columns='asin', values='overall').fillna(0)

    # Convert the dataframe to a matrix.
    # This matrix still contains NaN values.
    reviewer_product_sparse = csr_matrix(reviewer_product_dataframe.values)

    return (reviewer_product_sparse, reviewer_product_dataframe)

####################################################
#####     IMPLEMENTING k NEAREST NEIGHBORS     #####
####################################################

# Input a matrix and return a k_nn model using cosine similarity.
# NOTE: In the future, this should be switched to a centered cosine.
def k_nn(matrix):

    try:
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(matrix)
        return model_knn

    except:
        print('Please try another matrix.')
        return None


##########################################
#####     MAKING RECOMMENDATIONS     #####
##########################################

def make_recommendations(dataframe):
    query_index = np.random.choice(dataframe.shape[0])
    distances, indices = model_knn.kneighbors(dataframe.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(dataframe.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, dataframe.index[indices.flatten()[i]], distances
                                                           .flatten()[i]))
    return None



#####################################
#####     RUNNING FUNCTIONS     #####
#####################################

training_data = get_training_data('shortened_training_data.json')
# convert_to_csv(shortened_training_data, 'shortened_training_data.csv')
reviewer_product_sparse = create_reviewer_product_matrix(training_data)[0]
reviewer_product_dataframe = create_reviewer_product_matrix(training_data)[1]
model_knn = k_nn(reviewer_product_sparse)
make_recommendations(reviewer_product_dataframe)


####################################
#####     PRINT STATEMENTS     #####
####################################

# print(shortened_training_data[['reviewerID', 'asin', 'reviewerName', 'helpful', 'overall', 'summary']].head())
# print(shortened_training_data['reviewerID'].value_counts())
# print(shortened_training_data['asin'].value_counts())
# print(reviewer_product_matrix)
# print(reviewer_product_dataframe)
# print(model_knn)