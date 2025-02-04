import keras.applications
import keras.datasets
import keras.datasets.mnist
import keras.losses
import keras.metrics
import keras.optimizers
import keras.utils
import tensorflow as tf
import keras
import PIL
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

# Model definition

base_model = keras.applications.ResNet50(
    input_shape= (224,224,3),
    weights= "imagenet",
    include_top= False                      # Removes the final fully-connected layer
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation= "relu")(x)
predictions = layers.Dense(8)(x)

model = keras.Model(base_model.input, predictions)

model.compile(
    optimizer= keras.optimizers.Adam(learning_rate= 0.001),
    loss = keras.losses.CategoricalCrossentropy(from_logits= True),
    metrics= ["accuracy"]
)


# Data preprocessing

train_dir = "data/train"
val_dir = "data/val"

train_datagen = ImageDataGenerator(
    rescale= 1.0/255,
    rotation_range= 20,
    validation_split= 0.15
)

test_datagen = ImageDataGenerator(
    rescale = 1.0/255
)

train_generator = train_datagen.flow_from_directory(
    directory= train_dir,
    target_size= (224,224),
    batch_size= 64,
    subset= "training",
    shuffle= True,
    color_mode= "rgb",
    class_mode= "categorical"
)

val_generator = train_datagen.flow_from_directory(
    directory= train_dir,
    target_size= (224, 224),
    batch_size= 64,
    subset= "validation",
    class_mode= "categorical"
)


test_generator = test_datagen.flow_from_directory(
    directory = val_dir,
    target_size= (224,224),
    batch_size= 64,
    class_mode= "categorical"
)
# Training

history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // train_generator.batch_size,
    shuffle= True,
    epochs = 2,
    verbose= 1
)

_ ,test_scores = model.evaluate(test_generator, verbose = 2)

print("Test Accuracy: ", test_scores)

