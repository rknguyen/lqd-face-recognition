'''
created by rknguyen
'''

import os
import numpy as np
from datetime import datetime
from sklearn import neighbors

embeds_path = './embeds'


def now():
    return datetime.now().strftime('%Y%m%d%H%M%S')


def folder_resolver(user_id):
    return f'{embeds_path}/{user_id}'


def filename_resolver():
    return f'{now()}.npy'


def add_embeds(embeds, user_id):
    folder, filename = folder_resolver(user_id), filename_resolver()
    if not(os.path.exists(folder)):
        os.mkdir(folder)
    np.save(os.path.join(folder, filename), embeds)


def load_embeds_by_id(user_id):
    embeds = []
    folder = folder_resolver(user_id)
    for name in os.listdir(folder):
        embeds.append(np.load(os.path.join(folder, name))[0])
    return embeds


def load_all_embeds():
    embeds = []
    for user_id in os.listdir(embeds_path):
        embeds.append([user_id, load_embeds_by_id(user_id)])
    return embeds


def knn_init():
    clf = neighbors.KNeighborsClassifier(n_neighbors=1)

    train_X = []
    train_Y = []
    all_embeds = load_all_embeds()
    for user_id, user_embeds in all_embeds:
        for embed in user_embeds:
            train_X.append(embed)
            train_Y.append(user_id)

    clf.fit(train_X, train_Y)

    return clf


if __name__ == '__main__':
    # knn_init()
    pass
