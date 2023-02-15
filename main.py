# Importing all necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, \
    Dense, Flatten, UpSampling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.applications as ka
from keras import models
from keras.optimizers import SGD, Adam
import cv2
import matplotlib.pyplot as plt
import pickle


# code source: https://www.youtube.com/watch?v=LKMi8Daf2ts
# loading data
def load_data(IMAGE_SIZE):
    DIRECTORY = r"/Users/yaolunyin/PycharmProjects/STAT940W2023DC1New/data"
    CATEGORY = ["train", "test"]

    output = []

    # train labels
    given_labels = pd.read_csv("train_labels.csv")
    given_labels['label'] = given_labels['label'].apply(lambda x: str(x))
    given_labels['id'] = given_labels['id'].apply(lambda x: str(x) + ".jpg")

    for cate in CATEGORY:
        path = os.path.join(DIRECTORY, cate)
        print(path)
        names = []
        images = []
        labels = []


        print("loading {}".format(cate))


        for file in os.listdir(path):
            # get the path name of the image
            img_path = os.path.join(path, file)

            # open the img (we don't need to resize because the image is small enough)
            # the image size = 32*32*3
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)

            # append the image and its corresponding label
            names.append(file)
            images.append(image)
            label = int(given_labels.loc[given_labels['id'] == file]['label'])
            labels.append(label)
        images = np.array(images, dtype="float32")
        labels = np.array(labels, dtype="int32")
        output.append((names, images, labels))
    return output

# display 25 images from the image array with the corresponding labels
def display_examples(class_names, images, labels):
    figsize = (20, 20)
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Some examples", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = cv2.resize(images[i], figsize)
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(class_names[labels[i]])
    plt.show()

# display one image
def display_one(class_names, pred_labels, images, index):
    image = images[index]
    plt.imshow(image.astype(np.uint8))
    plt.xlabel(class_names[pred_labels[index]])
    plt.show()

# VGG16: work well for large, high-resolution images
# batch size 128, epoch 5 ----> accuracy 0.5
# too complex, I need to try some simple models
# Resnet50
# dense121
# pt: pretrained
def model_pt(name):
    model = Sequential()
    if name == 'VGG16':
        model.add(ka.VGG16(weights=None, include_top=False, input_shape=(32,32,3)))
    elif name == 'VGG19':
        model.add(ka.VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3)))
    elif name == 'RN50':
        model.add(ka.ResNet50(weights='imagenet', include_top=False, input_shape=(75,75,3)))
    elif name == 'RN152V2':
        model.add(ka.ResNet152V2(weights='imagenet', include_top=False, input_shape=(224,224,3)))
    elif name == 'Dense121':
        model.add(ka.DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3)))
    elif name == 'MN':
        model.add(ka.MobileNet(weights=None, include_top=False, input_shape=(32,32,3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # flat1 = Flatten()(model.layers[-1].output)
    # class1 = Dense(128, activation='relu')(flat1)
    # output = Dense(10, activation='softmax')(class1)
    # model = Model(inputs=model.inputs, outputs=output)
    opt= SGD(learning_rate=0.01, momentum=0.9)
    #opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"


    # idea: pretrain VGG16(with Imagenet weights) on a self-made dataset.
    # Then, train the model on the required dataset
    # finally, make the prediction

    # I used 5 classes from CIFAR10 and 5 classes from CIFAR100 to create the self-made dataset

    # pretrain a model
    model = model_pt('VGG16')
    # train the model on a self-made dataset. 5 classes are from CIFAR10 and the others are from CIFAR100
    with open('./pickle_data/self_train.pickle', 'rb') as f:
        (X_train_new, y_train_new) = pickle.load(f)
    with open('./pickle_data/self_test.pickle', 'rb') as f:
        (X_test_new, y_test_new) = pickle.load(f)

    # merge X_train and X_test
    X_train_new = np.concatenate([X_train_new, X_test_new], axis=0)
    y_train_new = np.concatenate([y_train_new, y_test_new], axis=0)
    X_train_new = X_train_new / 255.0
    X_test_new = X_test_new / 255.0
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0002,
    #                               patience=3, min_lr=0.0005)
    # fit data to the model
    # parameters
    batch_size = 64
    epochs = 10
    history = model.fit(X_train_new, y_train_new,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stop])  # , reduce_lr])
    # validate model
    # loss, accuracy = model.evaluate(X_train_new, y_train_new)
    # print("Test loss:", loss)
    # print("Test accuracy:", accuracy)
    # save model
    model.save("VGG16-self.h5")

    # train same dataset with LR=0.001
    opt = SGD(learning_rate=0.001, momentum=0.9)
    # opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_new, y_train_new,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stop])
    model.save("VGG16-LR001.h5")

    # train same dataset with LR=0.0005
    model = models.load_model("VGG16_LR001.h5")
    opt = SGD(learning_rate=0.0001, momentum=0.9)
    # opt = RMSprop(lr=2e-5)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_new, y_train_new,
                        batch_size=64,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stop])
    model.save("VGG16-LR0001.h5")



    # pre-train finished-------------------------------------

    # training starts
    #############################################################
    # class names
    # load model
    #model = models.load_model("VGG16-self.h5")
    class_names = ['deer', 'horse', 'car', 'truck', 'small mammal',
                   'flower', 'tree', 'aquatic mammal', 'fish', 'ship']
    class_names_label ={class_name : i for i, class_name in enumerate(class_names)}
    nb_classes = len(class_names)

    # load data
    # NOTE: test_labels is garbage!!! We don't have test_labels in this case
    #IMAGE_SIZE = (75, 75)
    #(train_names, train_images, train_labels), (test_names, test_images, test_labels) = load_data(IMAGE_SIZE)
    # save data into pickle objects so that i can reload it more efficiently
    # with open('train_tuple.pickle', 'wb') as f:
    #     pickle.dump((train_names, train_images, train_labels), f)
    # with open('test_tuple.pickle', 'wb') as f:
    #     pickle.dump((test_names, test_images), f)

    # reload train and test data
    with open('./pickle_data/train_tuple_32.pickle', 'rb') as f:
        (train_names, train_images, train_labels) = pickle.load(f)
    with open('./pickle_data/test_tuple_32.pickle', 'rb') as f:
        (test_names, test_images) = pickle.load(f)

    # make test ids
    test_id = list(map(lambda s: s[:-4], test_names))
    test_id = list(map(int, test_id))

    # shuffle train data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=1990)
    train_images, train_labels = shuffle(train_images, train_labels, random_state=43)
    train_images, train_labels = shuffle(train_images, train_labels, random_state=38)

    # display some examples
    #display_examples(class_names, train_images, train_labels)

    # normalize the training data
    X_train = train_images/255.0
    X_test = test_images/255.0
    y_train = train_labels

    # Data augmentation: create data generator
    # datagen = ImageDataGenerator(width_shift_range=0.1,
    #                              height_shift_range=0.1,
    #                              rotation_range=60,
    #                              horizontal_flip=True,
    #                              vertical_flip=True,
    #                              validation_split=0.2)
    # # prepare iterator
    # traingen= datagen.flow(X_train, y_train, batch_size=64, subset='training')
    # valigen = datagen.flow(X_train, y_train, batch_size=64, subset='validation')

    # Callback section
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5)


    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0002,
    #                               patience=3, min_lr=0.0005)

    # fit data to the model
    # parameters
    batch_size = 64
    epochs = 10
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stop])#, reduce_lr])

    # fit generator for data augmentation
    # history = model.fit_generator(traingen,
    #                               steps_per_epoch= len(traingen),
    #                               epochs=epochs,
    #                               validation_data=valigen,
    #                               validation_steps=len(valigen),
    #                               callbacks=[early_stop])

    # make predictions
    predictions = model.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)

    # display one example
    # the label is the predicted one
    for i in range(20):
         display_one(class_names, pred_labels, test_images, i)



    # make submission file
    res_df = pd.DataFrame({"id": test_id, "label": pred_labels})
    res_df.set_index("id", inplace=True)
    res_df = res_df.sort_index()

    # save the result
    res_df.to_csv("./Result/VGG16_final2.csv")


