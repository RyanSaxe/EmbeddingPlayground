import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, 
    Conv2DTranspose, 
    MaxPooling2D,
    UpSample2D,
    Flatten,
    Dropout,
    Dense,
)

class MLP(tf.Module):
    def __init__(self, n_classes, dropout=0.2, name=None):
        super().__init__(name=name)
        assert n_classes <= 16, "max number of classes is 16"
        self.drop = Dropout(dropout)
        self.p1 = Dense(128, activation='relu')
        self.p2 = Dense(64, activation='relu')
        self.p3 = Dense(32, activation='relu')
        self.p4 = Dense(16, activation='relu')
        self.prediction = Dense(n_classes, activation='softmax')

    def __call__(self, x):
        x = self.p1(x)
        x = self.drop(x)
        x = self.p2(x)
        x = self.drop(x)
        x = self.p3(x)
        x = self.drop(x)
        x = self.p4(x)
        x = self.drop(x)
        return self.prediction(x)

class AutoEncoder(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # operations
        self.pool_2x2 = MaxPooling2D((2,2), padding='same')
        self.upsample_2x2 = UpSample2D((2,2))
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
        x = self.upsample_2x2(x)
        x = self.d2(x)
        x = self.upsample_2x2(x)
        x = self.d3(x)
        x = self.upsample_2x2(x)
        return self.reconstruction(x)

class AE_w_predicter(tf.Module):
    def __init__(self, adversarial=False, name=None):
        super().__init__(name=name)
        self.AE = AutoEncoder()
        self.predicter = MLP()
        self.adversarial = adversarial
        self.compiled = False

    def _compile(self):
        # default compillation
        self.compiled = True
        self.pred_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
        #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
        representation = self.flatten(encoded)
        class_prediction = self.predicter(representation)
        return decoded, class_prediction

    def _step(self, batch):
        assert self.compiled, 'cannot take a step without setting an optimizer and loss function'
        input_imgs, classes = batch
        with tf.GradientTape as tape:
            reconstruction, prediction = self.__call__(input_imgs)
            pred_loss = self.pred_loss(classes, prediction)
            ae_loss = self.ae_loss(input_imgs, reconstruction)
            loss = self.ae_lambda * ae_loss
            if self.adversarial:
                # want to make the predictors task impossible
                adv_pred_loss = self.pred_lambda/(pred_loss + 1e-10)
                loss = loss + adv_pred_loss
        ae_grads = tape.gradient(loss, self.AE.trainable_variables)
        pred_grads = tape.gradient(pred_loss, self.predicter.trainable_variables)
        self.ae_optim.optimizer.apply_gradients(zip(ae_grads, self.AE.trainable_variables))
        self.pred_optim.optimizer.apply_gradients(zip(pred_grads, self.predicter.trainable_variables))
        return loss

    def train(self, images, labels, batch_size, n_epochs):
        pass
