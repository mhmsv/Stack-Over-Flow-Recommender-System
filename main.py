import operator

import numpy as np
import pandas as pd

from config import TAGS, NUM_OF_SIMILAR_TAGS, DATASET, PAIRED_TAGS, MOST_SIMILAR_PAIRS
from utils import create_tag_question_dict, extract_question_set, extract_tag_set


def find_most_similar_tags(query):
    related_questions = tag2question[query]
    frequently_tags = dict()
    for question_id in related_questions:
        related_tags = question2tag[question_id]
        for tag in related_tags:
            if tag != query:
                if tag in frequently_tags:
                    frequently_tags[tag] += 1
                else:
                    frequently_tags[tag] = 1

    sorted_tags = dict(sorted(frequently_tags.items(), key=operator.itemgetter(1), reverse=True))
    most_frequent_tags = list(sorted_tags.keys())[:NUM_OF_SIMILAR_TAGS]
    return most_frequent_tags


def cal_euclidean_dist(tag_1, tag_2):
    size = len(questions_set)

    question_list_1 = np.array([0] * size)
    question_list_2 = np.array([0] * size)

    tag_1_questions = tag2question[tag_1]
    tag_2_questions = tag2question[tag_2]

    questions_ind_1 = np.array([i for i in range(size) if questions_set[i] in tag_1_questions])
    questions_ind_2 = np.array([i for i in range(size) if questions_set[i] in tag_2_questions])

    question_list_1[questions_ind_1] = 1
    question_list_2[questions_ind_2] = 1

    dist = np.linalg.norm(question_list_1 - question_list_2)
    dist = dist / np.sqrt(size)

    return dist


if __name__ == '__main__':
    data_frame = pd.read_csv(DATASET, sep='\t', na_filter=False)
    tag2question, question2tag = create_tag_question_dict(data_frame)
    questions_set = extract_question_set(data_frame)
    tags_set = extract_tag_set(data_frame)
    print('Data is prepared successfully!')

    print('\n---------------------------------------------------- Question 1 ----------------------------------------------------\n')
    for item in TAGS:
        most_similar_tags = find_most_similar_tags(item)
        print('Top 5 most related tags for {0} are : '.format(item), most_similar_tags)

    print('\n---------------------------------------------------- Question 2 ----------------------------------------------------\n')
    for tag1, tag2 in PAIRED_TAGS:
        distance = cal_euclidean_dist(tag1, tag2)
        print('The Euclidean distance between {0} and {1} is: '.format(tag1, tag2), distance)

    print('\n---------------------------------------------------- Question 3 ----------------------------------------------------\n')
    pair_tag_dist = dict()
    for tag1 in tags_set:
        for tag2 in tags_set:
            if tag1 != tag2 and not ((tag1, tag2) in pair_tag_dist) and not ((tag2, tag1) in pair_tag_dist):
                distance = cal_euclidean_dist(tag1, tag2)
                pair_tag_dist[(tag1, tag2)] = distance

    sorted_pairs = dict(sorted(pair_tag_dist.items(), key=operator.itemgetter(1), reverse=False))
    most_similar_pairs = list(sorted_pairs.keys())[:MOST_SIMILAR_PAIRS]
    print(most_similar_pairs)
