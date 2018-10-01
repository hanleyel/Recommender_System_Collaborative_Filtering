import pandas as pd
import json
import gzip
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
import csv

#######################################
#####     UNZIPPING JSON DATA     #####
#######################################

def unzip_json(filename):
    unzipped_data = pd.read_json(gzip.open(filename))
    return unzipped_data


########################################################
#####     LOADING AND WRITING JSON DATA TO CSV     #####
########################################################

# Output json training data as a Pandas dataframe.
def json_to_df(file_name):

    try:
        training_data = pd.read_json(file_name, lines=True)
        return training_data
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




#################################################################
#####     LOADING AND STORING CSV DATA AS SPARSE MATRIX     #####
#################################################################

# Returns dictionaries with unique users and products as keys and unique ints as values.
def create_user_product_dicts(filename):

    user_dict = {}
    product_dict = {}

    with open(filename, 'r') as train_file:
        file_reader=csv.reader(train_file, delimiter=',')
        next(file_reader, None)
        count_user = 0
        count_product = 0
        for row in file_reader:
            if row[0] not in user_dict:
                user_dict[row[0]] = count_user
                count_user += 1
            if row[1] not in product_dict:
                product_dict[row[1]] = count_product
                count_product += 1

    return user_dict, product_dict



def readUrm(filename, user_dict, product_dict):

    num_user_ids = len(user_dict)
    print(num_user_ids)
    num_product_ids = len(product_dict)
    print(num_product_ids)

    urm = np.zeros(shape=(num_user_ids, num_product_ids), dtype=np.float32)
    with open(filename, 'r') as trainFile:
        urmReader = csv.reader(trainFile, delimiter=',')
        next(urmReader, None)
        for row in urmReader:
            urm[user_dict[row[0]], product_dict[row[1]]] = float(row[2])

    return csr_matrix(urm, dtype=np.float32)

########################################################
#####     CONVERTING DATAFRAME TO A CSR MATRIX     #####
########################################################

# Pivot datarame and create Reviewer x Product matrix populated with ratings. Return tuple with sparse matrix and dataframe.
def create_reviewer_product_matrix(dataframe):

    # Pivot the dataframe so that unique reviewers are on the y axis and unique products are on the x axis.
    # NOTE: Removed zeros in order to perform algebraic operations.
    reviewer_product_dataframe = dataframe.pivot(index='asin', columns='reviewerID', values='overall').fillna(0)
    # reviewer_product_dataframe = reviewer_product_dataframe.fillna(reviewer_product_dataframe.mean())

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


# Make k Nearest Neighbors recommendations
def make_recommendations(dataframe):
    query_index = np.random.choice(dataframe.shape[0])
    distances, indices = model_knn.kneighbors(dataframe.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Items most similar to {0}:\n'.format(dataframe.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, dataframe.index[indices.flatten()[i]], distances
                                                           .flatten()[i]))
    return None


###########################################
#####     IMPLEMENTING SVD MATRIX     #####
###########################################

# De-mean the data
def de_mean(reviewer_product_dataframe):
    rp_matrix = reviewer_product_dataframe.values
    user_ratings_mean = np.mean(rp_matrix, axis = 1)
    r_de_meaned = rp_matrix - user_ratings_mean.reshape(-1, 1)

    return r_de_meaned, user_ratings_mean

def create_svd(de_meaned_matrix):
    U, sigma, Vt = svds(de_meaned_matrix, k=50)
    return U, sigma, Vt

def to_diagonal_matrix(sigma):
    sigma = np.diag(sigma)
    return sigma

# Make predictions from decomposed matrices
def re_compose_matrices(U, sigma, Vt, user_ratings_mean, reviewer_product_dataframe):
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = reviewer_product_dataframe.columns)
    return preds_df


######################################################
#####     MAKING PREDICTIONS FROM SVD MATRIX     #####
######################################################

# Merge datasets and calculate predicted ratings?


#####################################
#####     RUNNING FUNCTIONS     #####
#####################################

# unzipped_data = unzip_json('reviews.training.json.gz')
# training_data = json_to_df('reviews.training.json')

# convert_to_csv(training_data[['reviewerID', 'asin', 'overall']], 'reviews.training.csv')
# reviewer_product_matrix = create_reviewer_product_matrix(training_data)
# reviewer_product_sparse = reviewer_product_matrix[0]
# reviewer_product_dataframe = reviewer_product_matrix[1]
user_dict = create_user_product_dicts('reviews.training.csv')
print(user_dict[0])
print(user_dict[1])
sparse_matrix = readUrm('reviews.training.csv', user_dict[0], user_dict[1])
# model_knn = k_nn(reviewer_product_sparse)
# make_recommendations(reviewer_product_dataframe)
# de_meaned_matrix = de_mean(reviewer_product_matrix[1])
# rp_svd = create_svd(de_meaned_matrix[0])
# diag_matrix = to_diagonal_matrix(rp_svd[1])
# recomposed_matrix = re_compose_matrices(rp_svd[0], rp_svd[1][0], rp_svd[2], de_meaned_matrix[1],
#                                         reviewer_product_dataframe)
# convert_to_csv(recomposed_matrix, 'recomposed_matrix.csv')


####################################
#####     PRINT STATEMENTS     #####
####################################

# print(training_data)
# print(training_data.shape)
# print(shortened_training_data[['reviewerID', 'asin', 'reviewerName', 'helpful', 'overall', 'summary']].head())
# print(shortened_training_data['reviewerID'].value_counts())
# print(shortened_training_data['asin'].value_counts())
# print(reviewer_product_sparse)
# print(reviewer_product_dataframe)
# print(model_knn)
# print(de_meaned_matrix[0])
# print(rp_svd)
# print(rp_svd[0].shape)
# print(rp_svd[1].shape)
# print(rp_svd[2].shape)
# print(diag_matrix)
# print(recomposed_matrix)
# print(user_dict[0])
# print(user_dict[1])
print(sparse_matrix)