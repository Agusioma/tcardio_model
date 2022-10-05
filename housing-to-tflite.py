import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

file_path = 'datasets/heart.csv'
heart_data = pd.read_csv(file_path)

imputer = SimpleImputer(strategy="median")
proc_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

heart_data_proc = proc_pipeline.fit_transform(heart_data)
heart_data_t = heart_data.drop("output", axis=1)
heart_data_z = imputer.fit_transform(heart_data_t)

train_set, test_set = train_test_split(heart_data, test_size=0.2, random_state=42)
X_train_full, X_test, y_train_full, y_test = train_test_split(heart_data_z, heart_data.output, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

class Model(tf.Module):
    def __init__(self):
        input = tf.keras.layers.Input(shape=X_train.shape[1:])
        hidden1 = tf.keras.layers.Dense(50, activation="sigmoid")(input)
        hidden2 = tf.keras.layers.Dense(50, activation="sigmoid")(hidden1)
        #concat = keras.layers.concatenate([input, hidden2])
        output = tf.keras.layers.Dense(1)(hidden2)
        self.model = keras.models.Model(inputs=[input], outputs=[output])
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=20,
                       validation_data=(X_valid, y_valid))

    @tf.function(input_signature=[
        tf.TensorSpec([None, 13], tf.float32),
        tf.TensorSpec([None, ], tf.float32),
    ])
    def train(self, x, y):
        '''with tf.GradientTape() as tape:
            prediction = self.model(x)
            lossy = self.model.loss(y, prediction)
        gradients = tape.gradient(lossy, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": lossy}
        return result'''

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,13], dtype=tf.float64)
    ])
    def pred(self,x):
        return self.model(X_test[0])

    @tf.function(input_signature=[
        tf.TensorSpec([None, 13], tf.float32),
    ])
    def predictee(self, x):
        predictions = self.model(x)
        return {
            "rent": predictions
        }

    @tf.function(input_signature=[
        tf.TensorSpec([None, 13], tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype, name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors


# NUM_EPOCHS = 100
BATCH_SIZE = 100
# epochs = np.arange(1, NUM_EPOCHS + 1, 1)
# losses = np.zeros([NUM_EPOCHS])
m = Model()
'''train_labels = tf.keras.utils.to_categorical(y_train)
test_labels = tf.keras.utils.to_categorical(y_test)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(BATCH_SIZE)
'''
'''for x, y in train_ds:
    result = m.train(x, y)'''
m
#%%
m.save('/tmp/model.ckpt')

SAVED_MODEL_DIR = "saved_model"
#%%
tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures={
        'predictee':
            m.predictee.get_concrete_function(),
        'save':
            m.save.get_concrete_function(),
        'restore':
            m.restore.get_concrete_function(),
    })
#%%
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

model_file_path = os.path.join('housing_model.tflite')
with open(model_file_path, 'wb') as model_file:
    model_file.write(tflite_model)