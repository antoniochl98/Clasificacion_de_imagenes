
import sys
import os
import pickle
target_dir = './modelo/'
option=1
while (option!=0):
    print("What do you want to do?\n1.Create a model\n2.Use a model\n0.Exit")
    option=int(input())
    if option==1:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras import optimizers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
        from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
        from tensorflow.keras import backend as K

        K.clear_session()
        #data_entrenamiento = './data/entrenamiento'
        print("Write the route of the training data:")
        data_entrenamiento=input()
        
        #data_validacion = './data/validacion'
        print("Write the route of the validation data:")
        data_validacion=input()
        model="prueba"
        print("Write the name of the model:")
        model=input()
        """
        Parameters
        """
        epocas=20
        longitud, altura = 150, 150
        batch_size = 32
        pasos = 1000
        validation_steps = 300
        filtrosConv1 = 32
        filtrosConv2 = 64
        tamano_filtro1 = (3, 3)
        tamano_filtro2 = (2, 2)
        tamano_pool = (2, 2)
        clases = len(os.listdir(data_entrenamiento))
        lr = 0.0004
    


        ##Preparamos nuestras imagenes

        entrenamiento_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
            data_entrenamiento,
            target_size=(altura, longitud),
            batch_size=batch_size,
            class_mode='categorical')

        validacion_generador = test_datagen.flow_from_directory(
            data_validacion,
            target_size=(altura, longitud),
            batch_size=batch_size,
            class_mode='categorical')

        
        if(True):
            cnn = Sequential()
            cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
            cnn.add(MaxPooling2D(pool_size=tamano_pool))

            cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
            cnn.add(MaxPooling2D(pool_size=tamano_pool))

            cnn.add(Flatten())
            cnn.add(Dense(256, activation='relu'))
            cnn.add(Dropout(0.5))
            cnn.add(Dense(clases, activation='softmax'))

            cnn.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Adam(lr=lr),
                        metrics=['accuracy'])



            cnn.fit_generator(
                entrenamiento_generador,
                steps_per_epoch=pasos,
                epochs=epocas,
                validation_data=validacion_generador,
                validation_steps=validation_steps)

            
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(target_dir+"/"+model):
                os.mkdir(target_dir+"/"+model)
            cnn.save(target_dir+"/"+model+"/modelo.h5")
            cnn.save_weights(target_dir+"/"+model+"/pesos.h5")
            with open(target_dir+"/"+model+"/clases.dat", "wb") as f:
                pickle.dump(entrenamiento_generador.class_indices, f)
            print(entrenamiento_generador.class_indices)
    elif option==2:
        import numpy as np
        from keras.preprocessing.image import load_img, img_to_array
        import tensorflow as tf
        import os

        
        longitud, altura = 150, 150
        modelo = './modelo/modelo4.h5'
        pesos_modelo = './modelo/pesos4.h5'
        print("Write the name of the model:")
        modelo=input()

        cnn = tf.keras.models.load_model(target_dir+modelo+"/modelo.h5")
        cnn.load_weights(target_dir+modelo+"/pesos.h5")
        with open(target_dir+modelo+"/clases.dat", "rb") as f:
            clases=pickle.load(f)
        clases={clases[i]:i for i in clases}
        def predict(file):
          x = load_img(file, target_size=(longitud, altura))
          x = img_to_array(x)
          x = np.expand_dims(x, axis=0)
          array = cnn.predict(x)
          result = array[0]
          answer = np.argmax(result)

          return answer

        print("What do you want to do?\n1.Test the model\n2.Clasify an image")
        op=int(input())

        if(op==1):
            count=0
            resultados={i:0 for i in clases}
            print("Write the path of the directory of the class you want to test: ")
            route=input();
            for dirpath, dirnames, filenames in os.walk(route):
                for image in filenames:
                    route=dirpath+"/"+image
                    answer=predict(route)
                    count+=1
                    resultados[answer]+=1
            if sys.platform=='win32':
                os.system('cls')
            else:
                os.system('clear')
            print("number of images:",count)
            for i in clases:
                print(" number of images clasified as ",clases[i],":",resultados[i])
        elif(op==2):
            print("Write the path of the image you want to clasify: ")
            route=input();
            answer=predict(route)
            print("The animal in the picture is a "+clases[answer])
    elif option!=0:
        print("Invalid option")


    