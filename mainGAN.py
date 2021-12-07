import tensorflow as tf # tensorflow
import tensorflow.keras as keras # high level api for tensorflow
import numpy as np # numpy for array manipulation
import matplotlib.pyplot as plt # for plotting

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data() # 28x28 greyscale images

learning_rate_dis = 0.005
learning_rate_gen = 0.01
batch_size = 16
epochs = 10
seed_size = 20

# data type int -> float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# scale data between 0 - 1
x_train /= 255
x_test /= 255

print("shape of x_train and x_test:")
print(x_train.shape)
print(x_test.shape)

# discriminator model
dis_input = keras.layers.Input(shape=(28, 28, 1))
dis_conv = keras.layers.Conv2D(16, 3, 2, activation="tanh", padding="same")(dis_input) # hidden dense
dis_conv = keras.layers.Conv2D(24, 3, activation="tanh", padding="same")(dis_conv) # hidden dense
dis_conv = keras.layers.Conv2D(32, 3, activation="tanh", padding="same")(dis_conv) # hidden dense
dis_conv = keras.layers.Conv2D(48, 3, 2, activation="tanh", padding="same")(dis_conv) # hidden dense
dis_conv = keras.layers.Conv2D(64, 3, activation="tanh", padding="same")(dis_conv) # hidden dense
dis_conv = keras.layers.Conv2D(80, 3, activation="tanh", padding="same")(dis_conv) # hidden dense
dis_conv = keras.layers.Flatten()(dis_conv)
dis_dense = keras.layers.Dense(128, activation="tanh")(dis_conv)
dis_output = keras.layers.Dense(1, activation="sigmoid")(dis_dense) # output (real or fake?)
dis_model = keras.Model(dis_input, dis_output)
dis_model.summary()

dis_model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate_dis), loss="binary_crossentropy", metrics=["accuracy"])

# generator model
gen_input = keras.layers.Input(shape=(seed_size))
gen_dense = keras.layers.Dense(128, activation="tanh")(gen_input)
gen_reshape = keras.layers.Reshape((8, 8, 2))(gen_dense)
gen_conv = keras.layers.Conv2D(64, 3, activation="tanh", padding="same")(gen_reshape)
gen_conv = keras.layers.Conv2D(48, 3, activation="tanh", padding="same")(gen_conv)
gen_conv = keras.layers.UpSampling2D()(gen_conv)
gen_conv = keras.layers.Conv2D(32, 3, activation="tanh", padding="same")(gen_conv)
gen_conv = keras.layers.Conv2D(24, 3, activation="tanh", padding="same")(gen_conv)
gen_conv = keras.layers.UpSampling2D()(gen_conv)
gen_conv = keras.layers.Conv2D(16, 3, activation="tanh", padding="valid")(gen_conv)
gen_output = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="valid")(gen_conv)

gen_model = keras.Model(gen_input, gen_output)
gen_model.summary()

dis_model.trainable = False # freezes the discriminators weights for the upcoming combined model (does not have any effect on the already compiled discriminator model)

# combined model (generator and discriminator concatenated)
com_input = keras.layers.Input(shape=(seed_size))
com_generator_image = gen_model(com_input)
com_discriminator_output = dis_model(com_generator_image)
com_model = keras.Model(com_input, com_discriminator_output)

com_model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate_gen), loss="binary_crossentropy", metrics=["accuracy"])

indicies = np.arange(len(x_train)) # datapoints we want to train with

recent_loss_com = 0
recent_loss_dis = 0

recent_accuracy_com = 0
recent_accuracy_dis = 0

fig = plt.figure() # for plotting later on

for m in range(epochs):
    print("epoch", m)
    counter = 0
    np.random.shuffle(indicies) # shuffle indicies before each epoch
    for i in range(0, len(indicies), batch_size):
        counter += 1

        take_n = min(batch_size, len(indicies) - i) # take the next n datapoints
        batch_indicies = indicies[i:i+take_n] # image indicies to train with in this batch

        seed = np.random.normal(0, 1, (take_n, seed_size)) # sample a batch of seeds from a normal distribution

        generated_images = gen_model.predict(seed) # let the model generate a batch of images

        evaluate_com = com_model.evaluate(seed, np.ones(take_n), verbose=0) # evaluate the generator by calling the combined models evaluate function
        evaluate_dis = dis_model.evaluate(generated_images, np.zeros(take_n), verbose=0) # evaluate the discriminator

        recent_loss_com += evaluate_com[0]
        recent_loss_dis += evaluate_dis[0]
        recent_accuracy_com += evaluate_com[1]
        recent_accuracy_dis += evaluate_dis[1]

        dis_model.train_on_batch(x_train[batch_indicies], np.ones(take_n)) # train the discriminator with a real image
        dis_model.train_on_batch(generated_images, np.zeros(take_n)) # train the discriminator with a generated image
        com_model.train_on_batch(seed, np.ones(take_n)) # train the generator

        if counter == 20: # show progress after every 20th epoch

            print(f"images seen in epoch {m}: {i + take_n}")
            print(f"log loss gen: {recent_loss_com/counter}")
            print(f"log loss dis: {recent_loss_dis/counter}")
            print(f"accuracy gen: {recent_accuracy_com/counter}")
            print(f"accuracy dis: {recent_accuracy_dis/counter}")

            # reset aggregated loss and accuracy
            recent_loss_com = 0
            recent_loss_dis = 0

            recent_accuracy_com = 0
            recent_accuracy_dis = 0

            # draw the first 16 generated images from the batch
            for n in range(16):
                img = (np.reshape(generated_images[n], (28, 28))*255).astype(int)
                plt.subplot(4, 4, n + 1)
                plt.imshow(img)

            plt.draw()
            plt.pause(0.5)
            fig.clear()

            counter = 0