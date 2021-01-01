import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, 
    Conv2DTranspose, 
    MaxPooling2D,
    UpSampling2D,
    Flatten,
    Dropout,
    Dense,
)
from tqdm import tqdm
import numpy as np
import pdb

def leaky_relu(x, alpha=0.3):
  return tf.keras.activations.relu(alpha=alpha)

class MLP(tf.Module):
    def __init__(self, n_classes, dropout=0.2, name=None):
        super().__init__(name=name)
        assert n_classes <= 16, "max number of classes is 16"
        self.drop = Dropout(dropout)
        self.p1 = Dense(64, activation=leaky_relu)
        self.p2 = Dense(32, activation=leaky_relu)
        self.p3 = Dense(16, activation=leaky_relu)
        self.prediction = Dense(n_classes, activation='softmax')

    def __call__(self, x):
        x = self.p1(x)
        x = self.drop(x)
        x = self.p2(x)
        x = self.drop(x)
        x = self.p3(x)
        return self.prediction(x)

class AutoEncoder(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # operations
        self.pool_2x2 = MaxPooling2D((2,2), padding='same')
        self.UpSampling_2x2 = UpSampling2D((2,2))
        # encoder layers
        self.e1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.e2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.e3 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.latent = Flatten()
        # decoder layers
        self.d1 = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')
        self.d2 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')
        self.d3 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')
        # 3 because RGB, if black and white image, replace first 3 with a 1
        self.reconstruction = Conv2D(3, (3, 3), activation='sigmoid', padding='same')

    def __call__(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encoder(self, x):
        x = self.e1(x)
        x = self.pool_2x2(x)
        x = self.e2(x)
        x = self.pool_2x2(x)
        return self.e3(x)

    def decoder(self, x):
        x = self.d1(x)
        x = self.UpSampling_2x2(x)
        x = self.d2(x)
        x = self.UpSampling_2x2(x)
        x = self.d3(x)
        return self.reconstruction(x)

class AE_w_predicter(tf.Module):
    def __init__(self, n_classes, adversarial=True, name=None):
        super().__init__(name=name)
        self.AE = AutoEncoder()
        self.predicter = MLP(n_classes=n_classes)
        self.adversarial = adversarial
        self.compiled = False

    def _compile(self):
        # default compillation
        self.compiled = True
        self.pred_loss = tf.keras.losses.CategoricalCrossentropy()
        self.pred_optim = tf.keras.optimizers.Adam()
        self.pred_lambda = 1.0
        self.ae_loss = tf.keras.losses.MeanSquaredError()
        self.ae_optim = tf.keras.optimizers.Adam()
        self.ae_lambda = 1.0

    def compile(
        self,
        pred_loss,
        pred_optim,
        pred_lambda,
        ae_loss,
        ae_optim,
        ae_lambda
    ):
        self.compiled = True
        self.pred_loss = pred_loss
        self.pred_optim = pred_optim
        self.pred_lambda = pred_lambda
        self.ae_loss = ae_loss
        self.ae_optim = ae_optim
        self.ae_lambda = ae_lambda

    def __call__(self, x):
        encoded = self.AE.encoder(x)
        decoded = self.AE.decoder(encoded)
        representation = self.AE.latent(encoded)
        class_prediction = self.predicter(representation)
        return decoded, class_prediction

    @property
    def _pred_loss_if_random(self):
        1.0/self

    def _step(self, input_imgs, classes):
        assert self.compiled, 'cannot take a step without setting an optimizer and loss function'
        with tf.GradientTape(persistent=True) as tape:
            reconstruction, prediction = self.__call__(input_imgs)
            #pdb.set_trace()
            pred_loss = self.pred_loss(classes, prediction)
            ae_loss = self.ae_loss(input_imgs, reconstruction)
            loss = self.ae_lambda * ae_loss
            if self.adversarial:
                # want to make the predictors task impossible
                adv_pred_loss = self.pred_lambda/(pred_loss + 1e-10)
                loss = loss + adv_pred_loss
        ae_grads = tape.gradient(loss, self.AE.trainable_variables)
        pred_grads = tape.gradient(pred_loss, self.predicter.trainable_variables)
        self.ae_optim.apply_gradients(zip(ae_grads, self.AE.trainable_variables))
        self.pred_optim.apply_gradients(zip(pred_grads, self.predicter.trainable_variables))
        return loss, ae_loss, pred_loss

    def train(self, images, labels, batch_size, n_epochs, normalize=1.0):
        size = labels.shape[0]
        order = np.arange(size)
        n_batches = size // batch_size + (size % batch_size > 0)
        for epoch in range(n_epochs):
            progress = tqdm(
                total = n_batches,
                desc = f'Epoch {epoch + 1}/{n_epochs}',
                unit = 'Batch'
            )
            np.random.shuffle(order)
            losses = []
            ae_losses = []
            pred_losses = []
            for i in range(n_batches):
                batch_idx = order[i * batch_size:(i+1) * batch_size]
                input_imgs = images[batch_idx]
                classes = labels[batch_idx]
                loss, ae_loss, pred_loss = self._step(input_imgs/normalize, classes)
                losses.append(loss)
                ae_losses.append(ae_loss)
                pred_losses.append(pred_loss)
                progress.set_postfix(
                    loss=np.average(losses),
                    ae=np.average(ae_losses),
                    pred=np.average(pred_losses)
                )
                progress.update(1)
            progress.close()
            
    def train_generator(self, generator, n_epochs, normalize=1.0):
        for epoch in range(n_epochs):
            progress = tqdm(
                total = len(generator),
                desc = f'Epoch {epoch + 1}/{n_epochs}',
                unit = 'Batch'
            )
            losses = []
            ae_losses = []
            pred_losses = []
            for i in range(len(generator)):
                input_imgs, classes = generator[i]
                loss, ae_loss, pred_loss = self._step(input_imgs/normalize, classes)
                losses.append(loss)
                ae_losses.append(ae_loss)
                pred_losses.append(pred_loss)
                progress.set_postfix(
                    loss=np.average(losses),
                    ae=np.average(ae_losses),
                    pred=np.average(pred_losses)
                )
                progress.update(1)
            generator.on_epoch_end()
            progress.close()


