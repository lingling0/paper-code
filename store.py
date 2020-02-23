import pickle

from param import current_path


def store_data(data, filename):
    # print(current_path + filename)
    with open(current_path + filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename):
    # print(current_path + filename)
    with open(current_path + filename, 'rb') as f:
        return pickle.load(f)
