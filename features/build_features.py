from num_occurrences import num_occurrences
from user_features import *
from pos_tag import get_ngram_postag_vector
from sentiment_StanfordNLP import get_sentiment_value
from constants import STANCE_LABELS_MAPPING
from emoticon import get_emoticons_vectors
from brown_cluster import brown_cluster
from has_word import contain_noswearing_bad_words, contain_acronyms, contain_google_bad_words
from regular_expressions import regex_vector
from get_score import get_vectors
from word_length import average_word_length, description_length
from named_entity import get_named_entity
from negation import get_average_negation

def collect_feature(tweet):
    """
        Collect a set of featrues from a tweet
    :param tweet: a json object representing a tweet
    :return: A vector represents all the feature
    """

    feature_vector = []

    # Whether the user has description or not.
    feature_vector += has_description(tweet['user'])

    # Length of user description in words
    feature_vector += [description_length(tweet['user'])]

    # Average length of a word
    feature_vector += [average_word_length(tweet['text'])]

    # Whether the user has enabled geo-location or not.
    feature_vector += geo_enabled(tweet['user'])

    # Whether the user is verified or not
    feature_vector += user_verified(tweet['user'])

    # Number of followers
    feature_vector += num_followers(tweet['user'])

    # Number of statuses of user
    feature_vector += originality_score(tweet['user'])

    # Role score
    feature_vector += role_score(tweet['user'])

    # Engagement score
    feature_vector += engagement_score(tweet)

    # Favourites score
    feature_vector += favorites_score(tweet)

    # Is a reply or not
    feature_vector += [1 if tweet['in_reply_to_status_id'] != None else 0]

    # Whether the tweet contain dot dot dot or not and number of dot dot dot
    dotdotdot_occurrences = num_occurrences(tweet['text'], r'\.\.\.')
    feature_vector += [1 if dotdotdot_occurrences > 0 else 0, dotdotdot_occurrences]
    # feature_vector += [1 if dotdotdot_occurrences > 0 else 0]

    # Whether the tweet contain exclamation mark or not and number of exclamation marks
    exclamation_mark_occurrences = num_occurrences(tweet['text'], r'!')
    feature_vector += [1 if exclamation_mark_occurrences > 0 else 0, exclamation_mark_occurrences]
    # feature_vector += [1 if exclamation_mark_occurrences > 0 else 0]

    # Whether the tweet contain question mark or not and number of question marks
    question_mark_occurrences = num_occurrences(tweet['text'], r'\?')
    feature_vector += [1 if question_mark_occurrences > 0 else 0, question_mark_occurrences]
    # feature_vector += [1 if question_mark_occurrences > 0 else 0]

    # Brown clusters
    brown_cluster_vector, has_url = brown_cluster(tweet['text'])
    feature_vector += brown_cluster_vector

    # Contain URL feature
    feature_vector += [1 if has_url else 0]

    # Postag features
    feature_vector += get_ngram_postag_vector(tweet['text'], 1)
    feature_vector += get_ngram_postag_vector(tweet['text'], 2)
    feature_vector += get_ngram_postag_vector(tweet['text'], 3)
    feature_vector += get_ngram_postag_vector(tweet['text'], 4)

    # Sentiment features
    # feature_vector += get_sentiment_value(tweet['text'])
    feature_vector += [0]

    # Stance features
    stance_vector = [0,0,0,0]
    feature_vector += stance_vector


    # Emoticon feature
    feature_vector += get_emoticons_vectors(tweet['text'])

    # Has Acronyms
    feature_vector += contain_acronyms(tweet['text'])

    # Has bad words
    feature_vector += contain_google_bad_words(tweet['text'])

    # Has no swearing bad words
    feature_vector += contain_noswearing_bad_words(tweet['text'])

    # Regex
    feature_vector += regex_vector(tweet['text'])

    # Doubt Score, No Doubt Score, Surprise score
    # feature_vector += get_vectors(tweet['text'])
    feature_vector += [0,0,0]

    # Get Named Entity Recognition
    feature_vector += get_named_entity(tweet['text'])

    # Negation
    feature_vector += get_average_negation(tweet['text'])


    return feature_vector
