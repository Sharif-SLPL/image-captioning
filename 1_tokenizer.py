from keras.preprocessing.text import Tokenizer
from pickle import dump


def load_doc(filename):
    """
    Load the doc file.

    :param filename: file name
    :type filename: str
    :return: file context as str
    """
    file = open(filename, 'r',encoding='utf-8')
    text = file.read()
    file.close()
    return text


def load_set(filename):
    """
    Load a pre-defined list of photo identifiers.

    :param filename: file name
    :type filename: str
    :return: list of photo identifiers
    """
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def load_clean_descriptions(filename, dataset):
    """
    Load clean descriptions.

    :param filename: file name
    :type filename: str
    :param dataset: list of photo identifiers
    :return: cleaned descriptions
    """
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


def to_lines(descriptions):
    """
    Convert a dictionary of clean descriptions to a list of descriptions.

    :param descriptions: descriptions
    :param descriptions: dict
    :return: list of descriptions
    """
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    """
    Fit a tokenizer given caption descriptions.

    :param descriptions: descriptions
    :param descriptions: list
    :return: the tokenizer object
    """
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


filename = 'train.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

tokenizer =  create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
