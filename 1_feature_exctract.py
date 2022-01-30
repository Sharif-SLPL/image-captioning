from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from os import listdir


def extract_features(directory):
    """
    Extract features from each photo in the directory.

    :param directory: directory in which photos are.
    :type directory: str
    :return: features as a dict
    """
    model = InceptionV3()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())

    # extract features from each photo
    features = dict()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(299, 299))

        image = img_to_array(image)
        # reshape
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features


directory = 'Flicker8k_Dataset/'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))

dump(features, open('features.pkl', 'wb'))
