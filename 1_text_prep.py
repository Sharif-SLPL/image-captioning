import string


def load_doc(filename):
    """
    Load the doc file.

    :param filename: file name
    :type filename: str
    :return: doc text
    """
    file = open(filename, 'r', encoding='utf8')
    text = file.read()
    file.close()
    return text


def load_descriptions(doc):
    """
    Extract descriptions for images.

    :param doc: document text
    :type doc: str
    :return: a mapping between image ids to image descriptions
    """
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    """
    Clean given descriptions.

    :param descriptions: given descriptions
    :param descriptions: dict
    :return: None
    """
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)


def to_vocabulary(descriptions):
    """
    Convert the loaded descriptions into a vocabulary of words.

    :param descriptions: image descriptions
    :type descriptions: dict
    :return: vocabulary of descriptions
    """
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


def save_descriptions(descriptions, filename):
    """
    Save the descriptions to a file, one per line.

    :param descriptions: image descriptions
    :type descriptions: dict
    :param filename: output filename
    :type filename: str
    :return: None
    """
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding='utf-8')
    file.write(data)
    file.close()


filename = 'Flickr8k_text/farsi_8k_human.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

clean_descriptions(descriptions)
vocabulary = to_vocabulary(descriptions)
print('Vocabulary Size: %d' % len(vocabulary))

save_descriptions(descriptions, 'descriptions.txt')
