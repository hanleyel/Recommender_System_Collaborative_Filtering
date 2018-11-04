# Ratings Imputation Using Singular Value Decomposition:
# Applied and Theoretical Implementations

Retailers and service providers offer an increasingly large number of goods and services to consumers. Instead of increasing sales, however, research shows that exposure to a wide variety of choices can actually paralyze potential customers. In fact, researchers have found that reducing the options consumers face can actually lead to increased sales and better overall satisfaction with a product. Organizations that provide a wide array of choices to their end users stand to benefit from tools that simplify users’ decision making.

One way to reduce the perceived complexity of choice without sacrificing access to options is to implement a recommender system. Based both on a user’s prior choices and the choices of other users, a recommender system calculates the likelihood that a user will give a high rating to a particular item. The system can then propose new items to the user based on the user’s inferred preferences.

This project explores the implementation of a recommender system that imputes ratings for missing data points in a sparse matrix based on singular value decomposition. In other words, the project aims to infer all possible rating values for each user and item combination, given a set of users who have each rated only a handful of items. This is accomplished via factorization of a user/item rating matrix in order to discover latent dimensions within the data. The user/item rating matrix is then reconstructed based on each user and item’s latent factors, resulting in a dense matrix with predicted ratings for each user/item pair.

### Applied Implementation
My first recommender system implementation is built on the scikit Surprise library for Python. Surprise provides an SVD algorithm that allows the user to customize their desired number of latent features. This proved to be an accurate and efficient method for imputing missing ratings data. Surprise also provides support for recommending items to users based on cosine similarity.

The Surprise implementation that I set up was very simple to use, and it output relatively accurate results. On my first Kaggle submission (after a dummy test submission) I obtained a score of 1.03 (root mean square error) with a k-value of 100. However, the downside of the simplicity of my implementation of the Surprise library is that there was less opportunity for customization given that I did not include any parameters beyond K for tuning the system. This meant that in my version of the implementation I was only able to change the K value, which resulted in little change among submissions (see Figure 4). In the future, if I were using Surprise for another project, I would be sure to implement the tuning parameters in order to output even better results.

### Theoretical Implementation
To gain a better understanding of how an SVD-based recommender system works behind the scenes, I implemented an SVD-based system built semi-from scratch in Python using Scipy’s linalg library as the engine for my SVD algorithm. Below is a discussion of my SVD-based recommender system implementation.

#### Json files to CSV
One of the biggest challenges in this project was working with a large dataset. The json training file was over 1.5 GB so reducing its size was a first priority. I decided to convert the dataset to CSV format, keeping only the most relevant data (reviewer ID, product ID, and Rating). This reduced the size of the data to about 37 M. Using CSV format also had the advantage of being easy to read into dataframes and matrices.

#### Dense Matrix Construction
I ultimately wanted to work with my data in matrix form. I first tried using pandas dataframes in order to accomplish this, as the dataframe option allowed me to easily map string-based reviewer and product IDs to the data. However, using pandas quickly proved to be an inefficient method for working with this large dataset. I switched to using a dense matrix as my first pass for uploading the data from CSV format. This allowed me to upload data with unique users along the y-axis, unique products along the x-axis, and users’ product ratings in the cell values (empty cells were filled with zeros).

#### Name - Index Dictionaries
While the reviewer and product IDs were stored as strings in the rating and test data, the ratings matrices had to be queried via integer indices. In order to map the string IDs onto matrix indices I created name-index dictionaries that mapped each unique reviewer ID and product ID onto an integer between zero and the length or width of the matrix.

#### Merged Test Data
One of the greatest challenges in implementing this recommender system was handling the “cold start” problem, which required the system to provide ratings for queries involving both unknown users and unknown items. I decided to handle this problem by merging the test data into the training data set, with products along the x-axis and users along the y-axis. I then weighted all cells by their row and column means (i.e., creating biases that could be fine-tuned if I were to eventually implement stochastic gradient descent).

#### Sparse Matrix Normalization
Before implementing the SVD algorithm I implemented a weighted global average row-wise and column-wise, then normalized the data by zero-centering it. I used a sparse matrix to accomplish this because working with a dense matrix was prohibitively memory-intensive at this point.

#### SVD Implementation
I tried four different SVD implementations: The Numpy library’s svd model, the Scipy linalg library’s svd model, the sparsesvd model, and the Scipy linalg library’s sparse.svds model.

I first tried sparsesvd with the intention of implementing the SVD algorithm based on sparse matrices for efficiency purposes. However, there were two issues with this implementation: First, the sparsesvd implementation returned odd U, S, and Vt values. Despite spending several hours troubleshooting, I couldn’t manage to recompose the U, S, and Vt values in a meaningful way (Figure 3). My intuition is that the sparsesvd implementation applies some form of normalization to the matrix before factoring it. The sparsesvd documentation is ironically sparse itself, though, and I eventually gave up on that implementation.

#### Matrix Reconstruction
Matrix reconstruction was straightforward. I took the dot product of the U, S, and Vt factors that were obtained from my normalized matrix. After recomposing the data I then added the global average to the recomposed matrix to rescale to a five-point rating system. This strategy populated ratings predictions into the matrix for all user-product pairs.

#### Querying
Querying the recomposed matrices for specific predictions was relatively straightforward; I obtained the index of the user and movie pair using the name-index dictionaries, then queried the recomposed prediction matrix for that index.

### Evaluation
I ultimately was able to return an RMSE evaluation value of 1.138 with a shortened training file of 50000 observations and a development file of 5000 observations. This was an improvement over the baseline global mean RMSE value of 1.19, demonstrating that even with a small sample size my algorithm appears to be working. However, despite my optimization efforts, I was not able to run the data using the full test data file after I made a modification that corrected a weighting error but increased the program’s memory requirements. As of this writing, this has prevented me from submitting a final file to Kaggle. However, I did locally implement an RMSE evaluation to assess the algorithm using the development file.

### Next Steps

Moving forward, there are several ways I could improve my code:
-Better optimize to reduce processing bottlenecks. This could potentially include making better use of sparse matrices during SVD implementation.
-Simplify matrix querying. I believe there is redundancy that could be removed from my current matrix merge method.
-Implement stochastic gradient descent to fine-tune the number of latent factors I use (K value) and my bias terms for rows and columns.

### References
Becker, N. (2016). Matrix factorization for movie recommendations in Python. https://beckernick.github.io/matrix-factorization-recommender/
Leskovek, Rajamaran, and Ullman (2011). Mining of Massive Datasets: Dimensionality Reduction: Singular Value Decomposition. https://www.youtube.com/watch?v=dt9iJPNFqaI
Pyrathon, D. (2018). A practical guide to singular value decomposition in Python. https://www.youtube.com/watch?v=d7iIb_XVkZs
Vastel, A. (2016). A simple SVD recommender system using Python. https://antoinevastel.com/machine%20learning/python/2016/02/14/svd-recommender-system.html

# Running the Program
The codes for Option 1 and Option 2 are both saved in Jupyter notebook files. To run the program you will need to save the following six files to the same directory:

Recommender System Collaborative Filtering_Surprise_Library_Option_1.ipynb
Recommender_System_Collaborative_Filtering_Scipy_Option_2.ipynb
reviews.training.json.gz
reviews.test.unlabeled.csv
reviews.dev.json
reviews.dev.csv

Both programs are setup to run in Jupyter notebooks as-is. You can simply run each cell in sequential order and they will run using the specified training data. You may need to uncomment the first function call to unzip the JSON data.

Option 1 will split the training file in to a training and test portion and will output a labeled test file in the correct format for Kaggle.

Option 2 will use the specified training file to train the SVD model and, in its current configuration, will output a labeled dev file that can be run through the local RMSE analysis. As-is, this program uses only the first 10000 observations from the training file and 2000 observations from the dev file. These can be increased (see cells 8 and 9), but doing so will also increase run time. Running the program on 50000 observations from the training file and 5000 observations from the test file takes approximately one hour. The program will also output the dense matrices as files in .txt format and the  sparse matrices as files in .npz format. I was using these to reduce run time during development.

