import os.path as op
import pandas
from similarities import *
from typing import Callable
import heapq


def user(
        data_directory: str,
        number_of_recommendations: int,
        similarity_metric: Callable[[FloatValueDict, FloatValueDict], float],
        userId: int
):
    # #
    # load data and convert to a nested dict like
    # {user_x: {movie_i: rating_xi, ...}, ...}
    # #
        
    ratings_df = pandas.read_csv(
        op.join(data_directory, "ratings.csv"),
        usecols=["userId", "movieId", "rating"]
    )

    ratings: dict[int, FloatValueDict] = {}
    for row in ratings_df.itertuples(index=False):
        user, movie, rating = row.userId, row.movieId, row.rating
        ratings.setdefault(user, {}).update({movie: rating})


    # #
    # compute the similarity between the given user 
    # and each of all the other users
    # #

    all_similarities: FloatValueDict = {
        other_user: similarity_metric(
            ratings[userId], 
            ratings[other_user]
        ) for other_user in ratings if other_user != userId
    }


    # #
    # find movies not watched by the given user
    # #
    
    movies_df = pandas.read_csv(
        op.join(data_directory, 'movies.csv'),
        usecols=['movieId']
    )

    watched_movies = ratings_df[
        ratings_df['userId'] == userId
    ]['movieId']

    not_watched_movies = movies_df[
        (~ movies_df["movieId"].isin(watched_movies))
        & movies_df["movieId"].isin(ratings_df["movieId"])
    ]['movieId']


    # #
    # compute the recommendation score for each movie not watched by the given user, 
    # and return the top n
    # #

    def _score_of(not_watched_movie: int) -> float:
        watchers = ratings_df[
            ratings_df['movieId'] == not_watched_movie
        ]['userId']

        similarities: FloatValueDict = {
            watcher: all_similarities[watcher]
            for watcher in watchers
            # weighted average is defined only for non-negative weights
            if all_similarities[watcher] > 0  
        }

        k = 128
        if len(watchers) < k:
            return 0
        
        # for n = len(similarities) and k = 128 (n >> k), using the heapq.nlargest
        # method offers O(n*log(k)) time complexity, better than O(n*log(n)) of 
        # sorted(similarities, reversed=True)[k]
        top_k = dict(heapq.nlargest(
            k, similarities.items(), 
            key=lambda dict_item: dict_item[1]
        ))
        
        # evaluate the recommendation score formula for the given movie
        return sum([
            similarities[other_user] * ratings[other_user][not_watched_movie]
            for other_user in top_k
        ]) / sum([similarities[other_user] for other_user in top_k])


    scores: FloatValueDict = {
        not_watched_movie: _score_of(not_watched_movie)
        for not_watched_movie in not_watched_movies
    }

    # comment the following to return a list of the movie ids
    return dict(heapq.nlargest(
        number_of_recommendations, 
        scores.items(), 
        key=lambda dict_item: dict_item[1]
    ))
    
    top_n_movies: list[tuple[int, int]] = heapq.nlargest(
        number_of_recommendations, 
        scores.items(), 
        key=lambda dict_item: dict_item[1]
    )
    
    return [tupl[0] for tupl in top_n_movies]


def item(
        data_directory: str,
        number_of_recommendations: int,
        similarity_metric: Callable[[FloatValueDict, FloatValueDict], float],
        userId: int
):
    # #
    # load data and convert to nested dict like
    # {movie_i: {user_x: rating_ix, ...}, ...}
    # #
        
    ratings_df = pandas.read_csv(
        op.join(data_directory, "ratings"),
        usecols=["userId", "movieId", "rating"]
    )

    ratings: dict[int, FloatValueDict] = {}
    for row in ratings_df.itertuples(index=False):
        user, movie, rating = row.userId, row.movieId, row.rating
        ratings.setdefault(movie, {}).update({user: rating})
    

    # #
    # find movies not watched by the given user
    # #
    
    movies_df = pandas.read_csv(
        op.join(data_directory, 'movies.csv'),
        usecols=['movieId']
    )

    watched_movies = ratings_df[
        ratings_df['userId'] == userId
    ]['movieId']

    not_watched_movies = movies_df[
        (~ movies_df["movieId"].isin(watched_movies))
        & movies_df["movieId"].isin(ratings_df["movieId"])
    ]['movieId']


    # #
    # compute the recommendation score for each movie not watched by the given user, 
    # and return the top n
    # #

    def _score_of(not_watched_movie: int) -> float:
        similarities: FloatValueDict = {
            watched_movie: similarity_metric(
                ratings[not_watched_movie], 
                ratings[watched_movie]
            ) for watched_movie in watched_movies
        }

        similarities = {
            movie: similarities[movie]
            for movie in similarities
            if similarities[movie] > 0
        }

        k = 30
        if len(similarities) < k:
            return 0
        
        top_k = dict(heapq.nlargest(
            k, similarities.items(), 
            key=lambda dict_item: dict_item[1]
        ))
        
        # evaluate the recommendation score formula for the given movie
        return sum([
            similarities[movie] * ratings[movie][userId]
            for movie in top_k
        ]) / sum([similarities[movie] for movie in top_k])
    
    scores: FloatValueDict = {
        movie: _score_of(movie)
        for movie in not_watched_movies
    }

    # comment the following to return a list of the movie ids
    return dict(heapq.nlargest(
        number_of_recommendations, 
        scores.items(), 
        key=lambda dict_item: dict_item[1]
    ))
    
    top_n_movies: list[tuple[int, int]] = heapq.nlargest(
        number_of_recommendations, 
        scores.items(), 
        key=lambda dict_item: dict_item[1]
    )
    
    return [tupl[0] for tupl in top_n_movies]


def title(
        data_directory: str,
        number_of_recommendations: int,
        similarity_metric: Callable[[FloatValueDict, FloatValueDict], float],
        itemId: int
):
    movies_df = pandas.read_csv(
        op.join(data_directory, 'movies.csv'),
        usecols=['movieId', 'title']
    )
    
    DF: dict[int, int] = {}
    for title in movies_df['title']:
        for keyword in title.split():
            DF[keyword] = DF.get(keyword, 0) + 1

    TFIDF: dict[int, FloatValueDict] = {
        row.movieId: {
            keyword: row.title.count(keyword) / DF[keyword]
            for keyword in row.title.split()
        } for row in movies_df.itertuples()
    }

    similarities: FloatValueDict = {
        movie: similarity_metric(
            TFIDF[itemId], TFIDF[movie]
        ) for movie in TFIDF if movie != itemId
    }

    return dict(heapq.nlargest(
        number_of_recommendations,
        similarities.items(),
        key=lambda dict_item: dict_item[1]
    ))
