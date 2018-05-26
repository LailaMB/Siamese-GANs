"""""
This code uses GANs netwrok for domain adaptation
Loss for discriminator: categorical crossentropy
Loss for generator: matching distance + reconstruciton error 
"""""
from __future__ import print_function
import time
start_time = time.time()

import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.core import Lambda
import keras.backend as K
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.decomposition import PCA
from sklearn import preprocessing


np.random.seed(1337)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



# build generator
def build_generator(latent_size):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(128, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(128))
    model.add(Activation('sigmoid'))

    return model


# build decoder
def Build_Decoder(latent_size):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(128, input_dim=latent_size))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(4096))
    model.add(Activation('sigmoid'))
    return model




def build_discriminator(latent_disc,num_classes):


    input_data = Input(shape=(latent_disc,))

    Match_distribution = input_data  # This output will be used for feature matching
    x=Dense(128,input_dim=latent_disc, name='dense1')(input_data)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.5)(x)

    x=Dense(128,name='dense2')(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    x=Dropout(0.5)(x)

    prob_output=Dense(num_classes, activation='softmax')(x)

    Real_Fake=Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_data], outputs=[prob_output,Real_Fake,Match_distribution])
    model.summary()
    return model




# main code

if __name__ == '__main__':

    print('Code starts here: Cross-scence classification with GANs Network')
    np.random.seed(1)

    # Define some important paramters
    epochs=100
    batch_size=100
    match_feature_size=128
    latent_disc=128            # input to the discriminator
    latent_size=4096


    data = np.load('data/Toronto_Potsdam.npz')
    #data = np.load('data/Toronto_Trento.npz')
    #data = np.load('data/Toronto_Vaihingen.npz')
    #data = np.load('data/Vaihangen_Trento.npz')
    #data = np.load('data/Vaihingen_Potsdam.npz')
    #data = np.load('data/Trento_Potsdam.npz')



    Xtrain = data['X1']
    ytrain = data['y1']
    num_classes = ytrain.shape[1]
    Xtest = data['X1']
    ytest = data['y1']

    Xtest = preprocessing.normalize(Xtest)
    Xtrain = preprocessing.normalize(Xtrain)

    #build discriminator
    discriminator = build_discriminator(latent_disc,num_classes=num_classes)   # Create an instance of the discriminator (Classifier)
    discriminator.compile(loss=['categorical_crossentropy','binary_crossentropy','mse'],loss_weights=[1, 0, 0.],
                          optimizer=Adam(lr=0.0001))


    # build generator
    generator=build_generator(latent_size) # Create an instance of the generator
    latent1 = Input(shape=(latent_size,)) # Specify the size of the Source features to generator (4096, None: for variant batch size)
    latent2=Input(shape=(latent_size,))   # Specify the size of the Target features to generator (4096, None: for variant batch size)
    features1 = generator(latent1)   # Define the input and the output of the geneator Placeholder for the generator's Source output.
    features2 = generator(latent2)   #  Placeholder for the generator's Target output.

    # Add layer for training the generator

    # Build matching loss

    def MMD_distance(vects):
        x, y =vects
        h1 = K.mean(x, axis=0)
        h2 = K.mean(y, axis=0)
        return K.mean(K.abs(h1 - h2))  # (K.sum(K.square(x - y), axis=0, keepdims=True)) #K.sqrt(K.mean((x),axis=1)-K.mean((y),axis=1))

    def MMD_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1



    output_sim = Lambda(MMD_distance, output_shape=MMD_dist_output_shape)([features1, features2])
    #Similarity_Model.compile(loss='mse',optimizer=RMSprop(lr=0.0001))
    Similarity_Model=Model(inputs=[latent1,latent2], outputs=[output_sim])

    def MMD_loss(y_true, y_pred):
        return K.sum(y_pred)

    Similarity_Model.compile(loss=MMD_loss,loss_weights=[1],optimizer=Adam(lr=0.0001))


    #Build Reconstruction loss
    generator.trainable=True
    Decoder = Build_Decoder(latent_disc)
    Rec_latent2 = Decoder(features2)
    AE_Model = Model(inputs=latent2, outputs=Rec_latent2)  # Define the Generarotr -> Decoder model and its input and output


    ###########################################################################
    ## Change this value from 0, 0.2, 0.4, 0.6, 0.8, 1     # 1 is our main case which means equal contribution
    AE_Model.compile(loss='mse', loss_weights=[1],optimizer=Adam(lr=0.0001)) # Specify the loss and optimization for the discriminator
    ############################################################################


##############################################################
    # Classification Results without adaptation................................
    discriminator.trainable=False
    [main_output, aug_output, auxiliary_output] = discriminator(features1)
    combined_2 = Model(inputs=latent1, outputs=main_output)  # Define the combined model latent -> class label
    combined_2.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001))
    Real_Fake = np.zeros((Xtest.shape[0]))  # Trick for avoiding bugs in the code: not needed now
    match_Dist_RF = np.ones((Xtest.shape[0], match_feature_size))   # Trick for avoiding bugs in the code, return an intermediate output
    history = combined_2.fit(Xtrain,ytrain,batch_size=batch_size,epochs=epochs,verbose=2)
    post_prob = combined_2.predict(Xtest)
    Test_predict_before = np.argmax(post_prob,axis=1)
    actual_y = np.argmax(ytest,axis=1)
    cm_before_adaptation = ConfusionMatrix(actual_y, Test_predict_before)
    print('----> Before adaptation')
    print(cm_before_adaptation.stats_overall)
    print("--- %s Time in seconds without Adaptation---" % (time.time() - start_time))

################################################################


    discriminator = build_discriminator(latent_disc,num_classes=num_classes)  # Create another instance of the discriminator
    discriminator.compile(loss=['categorical_crossentropy', 'binary_crossentropy', 'mse'], loss_weights=[0, 1, 0.],
                          optimizer=Adam(lr=0.0001))


    # GAN leanring starts here

    for epoch in range(epochs):

        print('Epoch {} of {}'.format(epoch + 1, epochs))
        nb_batches = int(Xtrain.shape[0] / batch_size)

        nb_batches = int(Xtrain.shape[0] / batch_size)
        half_batch=int(batch_size/2)
        idx_train_shuffle = np.arange(Xtrain.shape[0])
        np.random.shuffle(idx_train_shuffle)


        epoch_gen_loss = []
        epoch_disc_loss = []


        index=0

        while index < nb_batches-1:

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half batch of features from source
            Source_data=Xtrain[idx_train_shuffle[index * half_batch:(index + 1) * half_batch]] #[0:50]  [50:100]
            Source_label=ytrain[idx_train_shuffle[index * half_batch:(index + 1) * half_batch]] # Select the same label
            Source_data_feat = generator.predict(Source_data)  #Get the Source output of feature from generator

            idx = np.random.randint(0, Xtest.shape[0],  half_batch)
            Target_data = Xtest[idx]  # select random target data
            Target_label=ytest[idx]
            Target_data_feat=generator.predict(Target_data)

            # Labels of source and target data second loss [1 0]:   1 real, Fake 0
            labels_RF = np.array([1] * Source_data.shape[0]   + [0] * half_batch)

            # Concatenate source and target data batches for training Discriminator
            X_batch= np.concatenate((Source_data_feat, Target_data_feat))

            # class labels for both source and target
            class_labels=np.concatenate((Source_label,Target_label),axis=0)

            # This will be used for the matching loss: omited at this step!!!!!!
            Not_used2=np.zeros((X_batch.shape[0], match_feature_size))

            # Train the discriminator Here:
            discriminator.trainable = True
            disc_loss=discriminator.train_on_batch(X_batch,[class_labels,labels_RF,Not_used2])
            epoch_disc_loss.append(disc_loss[0])


            # ---------------------
            #  Train Generator  Xtest = preprocessing.normalize(Xtest)
            #----------------------

            # Select a random batch of target features
            discriminator.trainable = False

            idx = np.random.randint(0, Xtrain.shape[0], half_batch)
            Source_data = Xtrain[idx]
            idx = np.random.randint(0, Xtest.shape[0], half_batch)
            Target_data = Xtest[idx]
            X_batch = np.concatenate((Source_data, Target_data))

            # Used for matching...................
            Not_used_3 = np.array([0] * half_batch)


            # Train the generator
            Loss1=Similarity_Model.train_on_batch([Source_data,Target_data],Not_used_3)
            #Loss1=0
            # Train decoder
            Loss2=AE_Model.train_on_batch(Target_data,Target_data)
            Loss3=AE_Model.train_on_batch(Source_data,Source_data)

            # Compute combined loss
            epoch_gen_loss.append(Loss1+Loss2+Loss3)

            Source_data=[]
            Source_label=[]
            index += 1
        print('\n[Loss_D: {:.3f}, Loss_G: {:.3f}]'.format(np.mean(epoch_disc_loss), np.mean(epoch_gen_loss)))



    # Classify target data and see the results


    # build discriminator
    discriminator = build_discriminator(latent_disc, num_classes=num_classes)
    discriminator.compile(loss=['categorical_crossentropy', 'binary_crossentropy', 'mse'], loss_weights=[1, 0, 0.],
                              optimizer=Adam(lr=0.0001))

    Xtest2 = generator.predict(Xtest)
    Xtrain2 = generator.predict(Xtrain)

    Real_Fake = np.zeros((Xtest.shape[0]))  # Trick for avoiding bugs in the code: not needed now
    match_Dist_RF = np.ones((Xtest.shape[0], match_feature_size))   # Trick for avoiding bugs in the code, return an intermediate output
    history = discriminator.fit(Xtrain2, [ytrain, Real_Fake,match_Dist_RF], batch_size=batch_size, epochs=100, verbose=2)
    [post_prob, Real_Fake,V] = discriminator.predict(Xtest2)
    Test_predicted = np.argmax(post_prob, axis=1)
    actual_y = np.argmax(ytest, axis=1)
    cm_after_adaptation = ConfusionMatrix(actual_y, Test_predicted)
    print('----> After adaptation')
    print(cm_after_adaptation.stats_overall)

    print('-----> Results summary.............')
    print('----> Before adaptation')
    print(cm_before_adaptation.stats_overall)

    print('----> After adaptation')
    print(cm_after_adaptation.stats_overall)

    print("--- %s Time in seconds After adaptation ---" % (time.time() - start_time))




    #########################################################
    #''"This part will be used to plot the results of features "
    ###########################################################

    # Work first on orignal data


    plt.figure(figsize=plt.figaspect(0.5))   # figure size
    plt.rcParams.update(plt.rcParamsDefault) # the default parameters
    X_batch = np.concatenate([Xtrain, Xtest], axis=0)

    # build autoencoder to map data to 2features only
    pca = PCA(n_components=2)
    pca.fit(X_batch)
    vals1 = pca.transform(Xtrain)
    vals2 = pca.transform(Xtest)
    # plot
    plt.subplot(121)
    plt.margins=0.5
    #plt.plot(vals1[:, 0], vals1[:, 1], 'b*')
    plt.plot(vals1[:,0], vals1[:,1],'b.', label='Source',ms=4)
    plt.plot(vals2[:, 0], vals2[:, 1],'r+', label='Target',ms=4)
    #plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.xlabel('PCA #1')
    plt.ylabel('PCA #2')
    plt.grid(False)
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2, mode="expand", borderaxespad=0.)  # plt.legend(loc='best', shadow=False)

    X_batch = np.concatenate([Xtrain2, Xtest2], axis=0)


    # build autoencoder to map data to 2features only
    pca = PCA(n_components=2)
    pca.fit(X_batch)
    vals1 = pca.transform(Xtrain2)
    vals2 = pca.transform(Xtest2)
    # plot\
    plt.subplot(122)
    plt.margins=0.5
    #plt.plot(vals1[:, 0], vals1[:, 1], 'b*')
    plt.plot(vals1[:,0], vals1[:,1],'b*', ms=4)
    plt.plot(vals2[:, 0], vals2[:, 1],'r+', ms=4)
    #plt.axis([-0.5, 0.5, -0.5, 0.5])
    plt.xlabel('PCA #1')
    plt.ylabel('PCA #2')
    plt.grid(False)

    #plt.show()
    plt.savefig('plots/P_Tr.png')




    ## This code is used for plotting the confucion matrix
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # P <-> V
    #classes_name=['Tree','Grass','House','Bare soil','Roads','Cars','Water','Solar panel']

    #Tor <-> V
    #classes_name=['Tree','Grass','House','Bare soil','Roads','Cars','Water','Solar panel','Train track']

    # Tr <-> Tor
    #classes_name=['Tree','Grass','House','Bare soil','Roads','Cars','Solar panel','Train track']

    #Tr <-> P
    #classes_name=['Tree','Grass','House','Bare soil','Roads','Cars','Solar panel']

    #Tor <-> P
    classes_name=['Tree','Grass','House','Bare soil','Roads','Cars','Water','Solar panel']


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Greens):
        # (Confusion matrix, Classes, Normalized or not, )
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=30)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        #plt.tight_layout()
        plt.ylabel('True label')
        plt.title('Predicted label')


    # Compute confusion matrix

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=classes_name,
    #                      title='Confusion matrix, without normalization')
    #plt.show()
    # Plot normalized confusion matrix
    plt.figure()
    cnf_matrix = confusion_matrix(actual_y, Test_predict_before) # Confusion matrix before adaptation
    np.set_printoptions(precision=2) # percesion 0.00
    plot_confusion_matrix(cnf_matrix, classes=classes_name, normalize=True,title='Predicted label')

    ## CHANGE FIGURE NAME
    plt.savefig('plots/P_Tr_NN.png')
    #plt.show()
    plt.figure()
    cnf_matrix = confusion_matrix(actual_y, Test_predicted)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=classes_name, normalize=True)

    ## CHANGE FIGURE NAME
    plt.savefig('plots/P_Tr_Aerial.png')