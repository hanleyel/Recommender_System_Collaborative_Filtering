# Recommender_System_Collaborative_Filtering

# About recommender systems (from Mining Massive Datasets): https://www.youtube.com/watch?v=h9gpufJFF-0&index=43&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV

# Implementing recommender systems (from Mining Massive Datasets): https://www.youtube.com/watch?v=6BTLobS7AU8&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=44

# kNN tutorial at this link: https://beckernick.github.io/music_recommender/

# SVD
# https://www.youtube.com/watch?v=dt9iJPNFqaI
# https://beckernick.github.io/matrix-factorization-recommender/
# https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c




A collaborative filtering recommender system to predict consumer ratings.

The basic idea of collaborative filtering is that we have a user:

USER X

to whom we want to make recommendations.

In order to do this, we will find a group of other users whose likes and dislikes are similar to those of USER X. (The neighborhood of USER X.) Once we find these users, we find other items that are liked by USER X's neighborhood and recommend those items to USER X.

The key is to find USER X's neighborhood. In order to do that, we need to define a notion of similarity between users.

How can we do that?

Create a vector representing each user's prior ratings.
We need to compare these vectors in order to determine similarity. But how do we deal with unknown values?

The intuition: Users with similar tastes have higher similarity than users with dissimilar tastes.

Option 1) Jaccard similarity: intersection of two users' ratings of a movie divided by the union of two users' ratings of a movie

However, Jaccard similarity doesn't take into account users who watched the same movie but rated the movie very differently (these users will appear to have a high similarity when in acutality their tastes are very different).

Problem: Jaccard similarity ignores rating values.

Option 2) Cosine similarity: Use cosine distance between the vectors. Similarity between A and B could be the angle between the cosine vectors, RA and RB.

In order to compute this, we have to insert values for the uknown ratings. The simplest thing to do is insert zeros.

This will marginally capture similarity in the right direction, but the magnitude isn't enough to communicate what we want.

The problem with cosine similarity is that it treats missing ratings as negative (uses 0 to fill in the blanks - the worst possible rating, assuming that people who haven't rated a movie would rate it as 0).

One way to fix cosine similarity is to use centered cosine. Subtract the row mean from each of the ratings. Blank values are treated as 0s. If you sum up the ratings in any row you will get 0. (0 becomes the average rating for each user. So positive ratings mean the user liked the movie more than average, negative mean they liked it less than average.)

Centered cosine captures intuition better and accounts for "tough raters" and "easy raters."

Centered cosine similarity is also known as the Pearson Correlation.

RATING PREDICTIONS

Option 1) Take the average rating of all users in the Neighborhood
(ignores the similarity values between users)

Option 2) Weight the average rating by similarity values (weighted average)


ITEM-ITEM COLLABORATIVE FILTERING
For item i, find other similar items

Typically outperforms user-user collaborative filtering
Items are "simpler" than users
Item similarity is more meaningful than user
