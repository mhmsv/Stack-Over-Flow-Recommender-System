import numpy as np


def extract_question_set(df):
    question_set = set()

    for i in range(len(df)):
        q_id = df.iloc[i, 0]
        question_set.add(q_id)

    return np.array(list(question_set))


def extract_tag_set(df):
    tags_set = set()

    for i in range(len(df)):
        tag = df.iloc[i, 1]
        if tag == 'nan':
            print('fffff')
        tags_set.add(tag)

    return np.array(list(tags_set))


def create_tag_question_dict(df):
    tag2question = dict()
    question2tag = dict()

    for i in range(len(df)):
        q_id = df.iloc[i, 0]
        tag = df.iloc[i, 1]

        if tag in tag2question:
            tag2question[tag].append(q_id)
        else:
            tag2question[tag] = [q_id]

        if q_id in question2tag:
            question2tag[q_id].append(tag)
        else:
            question2tag[q_id] = [tag]

    return tag2question, question2tag


if __name__ == '__main__':
    import pandas as pd
    from config import DATASET
    data_frame = pd.read_csv(DATASET, sep='\t')
    extract_tag_set(data_frame)
