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
        
    columns = ["user_id", "item_id", "rating"]
    ratings_df = pandas.read_csv(
        op.join(data_directory, "u.data"),
        sep='\t', names=columns, usecols=columns
    )

    ratings: dict[int, FloatValueDict] = {}
    for row in ratings_df.itertuples(index=False):
        user, movie, rating = row.user_id, row.item_id, row.rating
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
        op.join(data_directory, 'u.item'),
        sep='|', encoding='ISO-8859-1', 
        names=['item_id'], usecols=['item_id']
    )

    watched_movies = ratings_df[
        ratings_df['user_id'] == userId
    ]['item_id']

    not_watched_movies = movies_df[
        (~ movies_df["item_id"].isin(watched_movies))
        & movies_df["item_id"].isin(ratings_df["item_id"])
    ]['item_id']


    # #
    # compute the recommendation score for each movie not watched by the given user, 
    # and return the top n
    # #

    def _score_of(not_watched_movie: int) -> float:
        watchers = ratings_df[
            ratings_df['item_id'] == not_watched_movie
        ]['user_id']

        similarities: FloatValueDict = {
            watcher: all_similarities[watcher]
            for watcher in watchers
            # weighted average is defined only for non-negative weights
            if all_similarities[watcher] > 0  
        }

        k = 30
        if len(watchers) < k:
            return 0
        
        top_k = dict(sorted(
            similarities.items(), 
            key=lambda dict_item: dict_item[1],
            reverse=True
        )[:k])
        
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
    return dict(sorted(
        scores.items(), 
        key=lambda dict_item: dict_item[1],
        reverse=True
    )[:number_of_recommendations])
    
    top_n_movies: list[tuple[int, int]] = sorted(
        scores.items(), 
        key=lambda dict_item: dict_item[1],
        reverse=True
    )[:number_of_recommendations]
    
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
        
    columns = ["user_id", "item_id", "rating"]
    ratings_df = pandas.read_csv(
        op.join(data_directory, "u.data"),
        sep='\t', names=columns, usecols=columns
    )

    ratings: dict[int, FloatValueDict] = {}
    for row in ratings_df.itertuples(index=False):
        user, movie, rating = row.user_id, row.item_id, row.rating
        ratings.setdefault(movie, {}).update({user: rating})
    

    # #
    # find movies not watched by the given user
    # #
    
    movies_df = pandas.read_csv(
        op.join(data_directory, 'u.item'),
        sep='|', encoding='ISO-8859-1', 
        names=['item_id'], usecols=['item_id']
    )

    watched_movies = ratings_df[
        ratings_df['user_id'] == userId
    ]['item_id']

    not_watched_movies = movies_df[
        (~ movies_df["item_id"].isin(watched_movies))
        & movies_df["item_id"].isin(ratings_df["item_id"])
    ]['item_id']


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
        
        top_k = dict(sorted(
            similarities.items(), 
            key=lambda dict_item: dict_item[1],
            reverse=True
        )[:k])
        
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
    return dict(sorted(
        scores.items(), 
        key=lambda dict_item: dict_item[1],
        reverse=True
    )[:number_of_recommendations])
    
    top_n_movies: list[tuple[int, int]] = sorted(
        scores.items(), 
        key=lambda dict_item: dict_item[1],
        reverse=True
    )[:number_of_recommendations]
    
    return [tupl[0] for tupl in top_n_movies]


def title(
        data_directory: str,
        number_of_recommendations: int,
        similarity_metric: Callable[[FloatValueDict, FloatValueDict], float],
        itemId: int
):
    columns = ['item_id', 'title']
    movies_df = pandas.read_csv(
        op.join(data_directory, 'u.item'),
        sep='|', encoding='ISO-8859-1', 
        names=columns, usecols=columns
    )
    
    DF: dict[int, int] = {}
    for title in movies_df['title']:
        for keyword in title.split():
            DF[keyword] = DF.get(keyword, 0) + 1

    TFIDF: dict[int, FloatValueDict] = {
        row.item_id: {
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
