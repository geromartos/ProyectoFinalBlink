import sys # manipular archivos de la computadora
import os  # manipular archivos de la computadora
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # preprocesa las imagenes de entrenamiento
from tensorflow.python.keras import optimizers # optimiza el algoritmo
from tensorflow.python.keras.models import Sequential # nos permite ahcer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D #las capas para hacer las convulociones
from tensorflow.python.keras import backend as K # si hay una sesion de keras en segundo plano lo cierra

K.clear_session() #hace lo de la libreria K



data_entrenamiento = './data/entrenamiento' #variable data_entrenamiento almacena el lugar donde estan las imagenes 
data_validacion = './data/validacion' #variable data_validacion almacena el lugr de las imagenes

"""
Parameters
"""
epocas=20 # numero de veces que itineramos en las imagenes
longitud, altura = 150, 150 # tamaño de las imagenes
batch_size = 32 #numero de imagenes que procesa en cada uno de los pasos
pasos = 1000 # es el numero de veces que se procesa la informacion en cada epoca
validation_steps = 300 # al final de cada epoca se corre 300 pasos para ver como va el aprendizaje
filtrosConv1 = 32 #numero de filtros que aplicamos en cada convulucion 
filtrosConv2 = 64
tamano_filtro1 = (3, 3)#tamaño del filtro1, altura de 3 y logitud de 3
tamano_filtro2 = (2, 2)#del filtro 2
tamano_pool = (2, 2)
clases = 3 # tenemos 3 señales,giro,maxima y contramano
lr = 0.0004 # que tan grande van a ser los ajustes que hace la red, generalmente es un numero pequeño


##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(  # creamos un generador que dice como vamos a preprocesar la informacion
    rescale=1. / 255, # cada uno de nuestros pixeles pasa a tener un rango de 0-255 a 1-255, para ser mas eficiente
    shear_range=0.2, # genera imagenes inclinadas para que vea que la imgen no siempre esta horizontal o vertical
    zoom_range=0.2,  # a algunas imagenes les hace un pequeño zoom
    horizontal_flip=True) # va a tomar una imagen y la va invertir

test_datagen = ImageDataGenerator(rescale=1. / 255) # para la validacion, le queremos dar la imagen tal cual

entrenamiento_generador = entrenamiento_datagen.flow_from_directory( #ya generamos las imagenes para entrenar la red
    data_entrenamiento,# carga las imagenes de entrenamiento 
    target_size=(altura, longitud), #reescala a la  altura y longitud que definimos antes
    batch_size=batch_size, # es el batch que definimos antes
    class_mode='categorical') #la clasificacion es categorica,osea, con sus nombre(las señales)

validacion_generador = test_datagen.flow_from_directory( # lo mismo con las imagenes de validacion
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


###crear la red neuronal###

cnn = Sequential()# la red va a ser secuencial,varias capas apliadas entre ellas
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))# le decimos que nuestra primera capa va a tener el numero de filtros que ya pusimos antes,y la activacion va  ser relu
cnn.add(MaxPooling2D(pool_size=tamano_pool)) # 

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten()) #esa imagen que ahora es muy profunda pero muy chica, la vamos a hacer plana
cnn.add(Dense(256, activation='relu'))# despues de apanar la informacion, pasamos otro filtro
cnn.add(Dropout(0.5))# a esta capa densa, durante el entrenmiento le apagamos la mitad de las neuronas durante las rondas
cnn.add(Dense(clases, activation='softmax')) # otra capa densa

cnn.compile(loss='categorical_crossentropy', 
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy']) # le estamos diciendo que durante el entrenamiento que analice su entrenamiento


###entrenamiento##

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

target_dir = './modelo/' # guardamos modelo
if not os.path.exists(target_dir): # si no existe la carpeta modelo, la crea
  os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')# grabamos el modelo en un archivo
cnn.save_weights('./modelo/pesos.h5')