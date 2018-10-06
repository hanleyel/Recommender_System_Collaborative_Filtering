import pandas as pd
import json
import gzip
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import csv
from sparsesvd import sparsesvd
import math
import operator

#######################################
#####     UNZIPPING JSON DATA     #####
#######################################

def unzip_json(filename):
    print('Unzipping json file...')
    unzipped_data = pd.read_json(gzip.open(filename))
    return unzipped_data


########################################################
#####     LOADING AND WRITING JSON DATA TO CSV     #####
########################################################

# Output json training data as a Pandas dataframe.
def json_to_df(file_name):

    print('Converting json file to dataframe...')

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

    print('Converting dataframe to csv: ' + desired_filename + '...')

    try:
        return dataframe.to_csv(desired_filename, index=False)
    except:
        print('Please try another dataframe or file name.')

    return None


###############################################################################
#####     LOADING AND STORING CSV TRAINING DATA AS INDEX DICTIONARIES     #####
###############################################################################

# Returns dictionaries with unique users and products as keys and unique ints as values.
def create_user_product_dicts(filename):

    print('Creating dictionaries from CSV for unique users and products...')

    user_dict = {}
    product_dict = {}
    user_count = 0
    product_count = 0

    with open(filename, 'r') as train_file:
        file_reader=csv.reader(train_file, delimiter=',')
        next(file_reader, None)

        for row in file_reader:
            if row[0] not in user_dict:
                user_dict[row[0]] = user_count
                user_count += 1
            if row[1] not in product_dict:
                product_dict[row[1]] = product_count
                product_count += 1

    print('Len user and product dicts')
    print(len(user_dict))
    print(len(product_dict))
    # print(product_dict)

    print('Getting largest values')
    print('Users: ')
    lg_user = max(user_dict.items(), key=operator.itemgetter(1))[0]
    print(lg_user)
    print(user_dict[lg_user])
    print('Products: ')
    lg_prod = max(product_dict.items(), key=operator.itemgetter(1))[0]
    print(lg_prod)
    print(product_dict[lg_prod])

    print('User and items end counts')
    print('users:')
    print(user_count)
    print('products: ')
    print(product_count)

    print('dict lengths:')
    print('user')
    print(len(user_dict))
    print('products: ')
    print(len(product_dict))

    return user_dict, product_dict, user_count, product_count


###########################################################################
#####     LOADING AND STORING CSV TEST DATA AS INDEX DICTIONARIES     #####
###########################################################################

# Outputs dictionaries with unique test users and test products.
def get_test_users_products(filename, training_user_dict, training_product_dict, user_count, product_count):

    print('Importing test users and products...')

    test_user_count = user_count
    test_product_count = product_count
    test_user_dict = training_user_dict
    test_product_dict = training_product_dict

    with open(filename, 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter=',')
        next(test_reader, None)

        for row in test_reader:
            # Add unique users to test_user dictionary.
            if row[1] not in test_user_dict:
                test_user_dict[row[1]] = test_user_count
                test_user_count += 1
            # Add unique products to test_product dictionary.
            if row[2] not in test_product_dict:
                test_product_dict[row[2]] = test_product_count
                test_product_count += 1

    print('Len test dicts, user and product: ')
    print(len(test_user_dict))
    print(len(test_product_dict))
    print('Len test user and product counts: ')
    print(test_user_count)
    print(test_product_count)
    # print(test_user_dict)
    # print(test_product_dict)

    return test_user_dict, test_product_dict


############################################################
#####     CREATING DENSE MATRIX WITH TRAINING DATA     #####
############################################################

def training_mtx(filename, user_dict, product_dict):

        print('Creating a dense matrix from training data...')

        num_user_ids = len(user_dict)
        num_product_ids = len(product_dict)

        dense_matrix = np.zeros(shape=(num_user_ids, num_product_ids), dtype=np.float32)

        with open(filename, 'r') as train_file:
            matrix_reader = csv.reader(train_file, delimiter=',')
            next(matrix_reader, None)
            for row in matrix_reader:
                dense_matrix[user_dict[row[0]], product_dict[row[1]]] = float(row[2])

        print('training matrix shape: ')
        print(dense_matrix.shape)
        print(dense_matrix)

        return dense_matrix

############################################################################
#####     CREATING DENSE MATRIX WITH MERGED TRAINING AND TEST DATA     #####
############################################################################

def merged_mtx(test_file, train_file, test_user_dict, test_product_dict, user_dict,
               product_dict):

    print('Merging training and test data for ratings imputation...')

    num_user_ids = len(test_user_dict)
    num_product_ids = len(test_product_dict)

    merged_matrix = np.zeros(shape=(num_user_ids, num_product_ids), dtype=np.float32)

    with open(test_file, 'r') as test_file:
        file_reader=csv.reader(test_file, delimiter=',')
        next(file_reader, None)

        for row in file_reader:
            merged_matrix[test_user_dict[row[1]], test_product_dict[row[2]]] = float(0)

    with open(train_file, 'r') as train_file:
        file_reader=csv.reader(train_file, delimiter=',')
        next(file_reader, None)

        for row in file_reader:
            merged_matrix[user_dict[row[0]], product_dict[row[1]]] = float(row[2])

    print('Merged matrix tests: ')
    print(merged_matrix[[user_dict['A34DNO6UAH67Z0']],[product_dict['B000CDSS22']]]) # Should be 5
    print(merged_matrix[[user_dict['A3APW42N5MRVWT']], [product_dict['6305186774']]]) # Should be 2
    print(merged_matrix[[user_dict['A2M03PQV8R826Z']], [product_dict['B0000DKDUR']]]) # Should be 5

    return merged_matrix


########################################
#####     MATRIX NORMALIZATION     #####
########################################

def normalize_merged_matrix(matrix):

    print('Calculating ratings mean...')
    # matrix_mean = np.mean(matrix, axis=1)
    matrix_row_mean = np.true_divide(matrix.sum(1), (matrix != 0).sum(1))

    # matrix_mean_test = np.mean(matrix[matrix > 0])
    # print(matrix_mean_test)
    # matrix_mean = matrix.sum(1)/(matrix!=0).sum(1).astype(float)

    global_mean = np.mean(matrix)

    print('Normalizing the data...')
    normalized_matrix = matrix - matrix_row_mean.reshape(-1, 1)

    print('Test normalized mean value: ')
    print(matrix_row_mean[0])

    print('Fill nan with 0')
    print()


    return normalized_matrix, matrix_row_mean, global_mean

####################################################################
#####     LOADING AND STORING CSV DATA AS DE-MEANED MATRIX     #####
####################################################################

# def readUrm(filename, user_dict, product_dict):
#
#     print('Creating a first dense matrix from rating data...')
#
#     num_user_ids = len(user_dict)
#     num_product_ids = len(product_dict)
#
#     urm = np.zeros(shape=(num_user_ids, num_product_ids), dtype=np.float32)
#
#     with open(filename, 'r') as train_file:
#         urmReader = csv.reader(train_file, delimiter=',')
#         next(urmReader, None)
#         for row in urmReader:
#             urm[user_dict[row[0]], product_dict[row[1]]] = float(row[2])
#
#     print('Normalizing the matrix...')
#     urm_mean = np.mean(urm, axis=1)
#     urm_demeaned = urm - urm_mean.reshape(-1, 1)
#
#     print(urm_mean[0])
#
#
#     # print('Creating a sparse CSR matrix from dense rating matrix data...')
#     # urm_sparse_csr = scipy.sparse.csr_matrix(urm, dtype=np.float32)
#
#     # print('Creating a sparse CSC matrix from dense rating matrix data...')
#     # urm_sparse_csc = scipy.sparse.csc_matrix(urm, dtype=np.float32)
#
#
#     return num_user_ids, num_product_ids, urm_demeaned, urm_mean


####################################################
#####     IMPLEMENTING k NEAREST NEIGHBORS     #####
####################################################
#
# # Input a matrix and return a k_nn model using cosine similarity.
# # NOTE: In the future, this should be switched to a centered cosine.
# def k_nn(matrix):
#
#     try:
#         model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
#         model_knn.fit(matrix)
#         return model_knn
#
#     except:
#         print('Please try another matrix.')
#         return None
#
#
# # Make k Nearest Neighbors recommendations
# def make_recommendations(dataframe):
#     query_index = np.random.choice(dataframe.shape[0])
#     distances, indices = model_knn.kneighbors(dataframe.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
#     for i in range(0, len(distances.flatten())):
#         if i == 0:
#             print('Items most similar to {0}:\n'.format(dataframe.index[query_index]))
#         else:
#             print('{0}: {1}, with distance of {2}:'.format(i, dataframe.index[indices.flatten()[i]], distances
#                                                            .flatten()[i]))
#     return None


##############################################################
#####     IMPLEMENTING SVD MATRIX FROM SPARSE MATRIX     #####
##############################################################

# def computeSVD(sparse_matrix, K):
#
#     print('Computing SVD matrix...')
#
#     U, s, Vt = sparsesvd(sparse_matrix, K) #csr --> csc
#     S = np.diag(s)
#
#     # dim = (len(s), len(s))
#     # S = np.zeros(dim, dtype=np.float32)
#     # for i in range(0, len(s)):
#     #     S[i,i] = math.sqrt(s[i])
#
#     return U, S, Vt


################################################################################
#####     ALTERNATE IMPLEMENTING SVD MATRIX FROM DEMEANED DENSE MATRIX     #####
################################################################################

def compute_svd_from_demeaned(urm_demeaned):

    print('Computing svd from de-meaned matrix...')

    U, sigma, Vt = svds(urm_demeaned, k = 100)
    S = np.diag(sigma)

    return U, S, Vt


#####################################################################
#####     MAKING RATINGS PREDICTIONS FROM SPARSE SVD MATRIX     #####
#####################################################################

# def recompose_matrix(U, S, Vt, user_dict, product_dict, training_file, outfile):
#
#     rightTerm = np.dot(S, Vt)
#
#     print('Right Term')
#     print(rightTerm.shape)
#     print(rightTerm)
#
#     estimated_ratings = np.zeros(shape=(len(user_dict), len(product_dict)), dtype=np.float16)
#
#     with open(training_file, 'r') as test_file:
#         test_reader = csv.reader(test_file, delimiter=',')
#         next(test_reader, None)
#         with open(outfile, 'w') as outfile:
#             outfile_reader = csv.writer(outfile, delimiter=',')
#             outfile_reader.writerow(['userID', 'actual overall', 'predicted'])
#
#             for row in test_reader:
#                 pass
#
#                 print('U queried shape')
#                 u_queried = U[:, user_dict[row[0]]]
#                 # print(u_queried.shape)
#                 print(u_queried)
#                 #
#                 print('Right term queried shape')
#                 rt_queried = rightTerm[:, product_dict[row[1]]]
#                 # print(rt_queried.shape)
#                 print(rt_queried)
#                 #
#                 print('Product of u queried by rt queried')
#                 prod = np.dot(u_queried, rt_queried)
#                 print(prod.shape)
#                 print(prod)
#
#                 # print('Estimated ratings')
#                 # estimated_ratings[:, user_dict[row[0]]] = prod.todense()
#                 # print(estimated_ratings)
#
#                 # estimated_ratings[user_dict[row[0]], :] = prod.todense()
#                 # predicted_rating = (estimated_ratings[user_dict[row[0]], product_dict[row[1]]])
#                 outfile_reader.writerow([row[0], row[2], prod])
#
#     return estimated_ratings

###############################################################################
#####     ALTERNATE MAKE RATINGS PREDICTIONS FROM DEMEANED SVD MATRIX     #####
###############################################################################

def reconstruct_demeaned_matrix(U, S, Vt, urm_mean, testing_file, outfile, test_user_dict, test_product_dict):

    print('Reconstructing matrix and making predictions...')

    predicted_ratings = (np.dot(np.dot(U, S), Vt) + urm_mean.reshape(-1, 1))

    print('Predicted ratings')
    print(predicted_ratings)

    with open(testing_file, 'r') as test_file:
        test_reader = csv.reader(test_file, delimiter=',')
        next(test_reader, None)
        with open(outfile, 'w') as outfile:
            outfile_reader = csv.writer(outfile, delimiter=',')
            outfile_reader.writerow(['datapointID', 'overall'])

            for row in test_reader:

                prediction = predicted_ratings[test_user_dict[row[1]], [test_product_dict[row[2]]]]

                # if prediction[0] > 5:
                #     prediction[0] = 5
                #
                # if prediction[0] < 1:
                #     prediction[0] = 1

                outfile_reader.writerow([row[0], prediction[0]])

    return predicted_ratings[[user_dict['A3APW42N5MRVWT']], [product_dict['6305186774']]]



########################
#####     MAIN     #####
########################

# training_df = json_to_df('reviews.training.json')
# convert_to_csv(training_df.head(1000))
user_dict, product_dict, user_count, product_count = create_user_product_dicts('reviews.test.shortened.csv')
test_user_dict, test_product_dict = get_test_users_products('reviews.dev.shortened.csv', user_dict, product_dict,
                                                            user_count, product_count)
merged_matrix = merged_mtx('reviews.dev.shortened.csv', 'reviews.test.shortened.csv', test_user_dict, test_product_dict,
                           user_dict, product_dict)

# training_matrix = training_mtx('reviews.test.shortened.csv', user_dict, product_dict)

# print(merged_matrix)

normalized_matrix, matrix_mean, global_mean = normalize_merged_matrix(merged_matrix)

print(normalized_matrix)

# num_user_ids, num_product_ids, urm_demeaned, urm_mean = readUrm('reviews.test.shortened.csv', user_dict, product_dict)
# U, S, Vt = computeSVD(sparse_matrix, 3)
U, S, Vt = compute_svd_from_demeaned(normalized_matrix)

# model_knn = k_nn(reviewer_product_sparse)
# make_recommendations(reviewer_product_dataframe)

# estimated_ratings = recompose_matrix(U, S, Vt, user_dict, product_dict, 'reviews.training.csv', 'reviews.test.labeled.csv')

predicted_ratings = reconstruct_demeaned_matrix(U, S, Vt, matrix_mean, 'reviews.test.unlabeled.shortened.csv',
                                                'reviews.test.labeled.csv', test_user_dict, test_product_dict)
# print(predicted_ratings)


####################################
#####     PRINT STATEMENTS     #####
####################################
# Ideally, these could have been written as unit tests

# # Test user and product dictionaries
# print('Training user dict: ')
# print(user_dict['AMFIPCYDYWGVT'])
# print('(Should equal 0.)\n')
# print(user_dict['AT79BAVA063DG'])
# print('(Should equal 5332.)\n')
# print('Training product dict')
# print(product_dict['B0090SI56Y'])
# print('Should equal 0.)\n')
# print(product_dict['B0009UVCQC'])
# print('Should equal 19321.)\n')
#
# # Test test user and test product dictionaries
# print('Test user dict: ')
# print(test_user_dict['AT79BAVA063DG'])
# print('Should equal 5332.)\n')
# print('Test product dict: ')
# print(test_product_dict['B0009UVCQC'])
# print('Should equal 19321.)\n')
#
# print('Dense matrix')
# print(urm[user_dict['AMFIPCYDYWGVT'], product_dict['B0090SI56Y']])
# print('Should equal 4. \n')
#
# print('Sparse matrix: ')
# print(sparse_matrix)
# print('\n')
#
# print('Number of test user rows: ')
# print(num_user_ids)
# print('Should equal 123952.\n')
# print('Number of test product columns: ')
# print(num_product_ids)
# print('Should equal 50050.\n')
#
# print('U')
# print(U.shape)
# print(U)
# print('\n')
# print('S')
# print(S.shape)
# print(S)
# print('\n')
# print('Vt')
# print(Vt.shape)
# print(Vt)
# print('\n')