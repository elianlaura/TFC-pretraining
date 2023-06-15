# %% [markdown]
# # SimCLR Contrastive Training for Human Activity Recognition Tutorial

# %%
# Author: C. I. Tang
# Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
# Contact: cit27@cl.cam.ac.uk
# License: GNU General Public License v3.0


# %% [markdown]
# ## Imports

# %%
import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

# %%
# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# %%
# Library scripts
import raw_data_processing
import data_pre_processing   
import simclr_models
import simclr_utitlities
import transformations

# %%
working_directory = 'test_run/'
print(working_directory)

# %%
dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)


# %% [markdown]
# ### Downloading & Unzipping

# %%
import requests
import zipfile

# %%
#dataset_url = 'https://github.com/mmalekzadeh/motion-sense/blob/master/data/B_Accelerometer_data.zip?raw=true'

#r = requests.get(dataset_url, allow_redirects=True)
#with open(working_directory + 'B_Accelerometer_data.zip', 'wb') as f:
#    f.write(r.content)

# %%
#with zipfile.ZipFile(working_directory + 'B_Accelerometer_data.zip', 'r') as zip_ref:
#    zip_ref.extractall(working_directory)

# %% [markdown]
# ### Data Processing

# %%
accelerometer_data_folder_path = working_directory + 'B_Accelerometer_data'
user_datasets = raw_data_processing.process_motion_sense_accelerometer_files(accelerometer_data_folder_path)

# %%
with open(working_directory + 'motion_sense_user_split.pkl', 'wb') as f:
    pickle.dump({
        'user_split': user_datasets,
    }, f)

# %% [markdown]
# ## Pre-processing

# %%
# Parameters
window_size = 100
input_shape = (window_size, 6)

# Dataset Metadata
transformation_multiple = 1
dataset_name = 'motion_sense.pkl'
dataset_name_user_split = 'motion_sense_user_split.pkl'

label_list = ['null', 'sit', 'std', 'wlk', 'ups', 'dws', 'jog']
label_list_full_name = ['null', 'sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'jogging']
has_null_class = True

label_map = dict([(l, i) for i, l in enumerate(label_list)])

output_shape = len(label_list)

model_save_name = f"motionsense_acc"

sampling_rate = 50.0
unit_conversion = scipy.constants.g

# a fixed user-split
test_users_fixed = [1, 14, 19, 23, 6]
def get_fixed_split_users(har_users):
    # test_users = har_users[0::5]
    test_users = test_users_fixed
    train_users = [u for u in har_users if u not in test_users]
    return (train_users, test_users)


with open(dataset_save_path + dataset_name_user_split, 'rb') as f:
    dataset_dict = pickle.load(f)
    user_datasets = dataset_dict['user_split']


har_users = list(user_datasets.keys())
train_users, test_users = get_fixed_split_users(har_users)
print(f'Testing: {test_users}, Training: {train_users}')



# Here we convert the original dataset into a windowed one, and split it into training, validation and testing sets.
'''
np_train, np_val, np_test = data_pre_processing.pre_process_dataset_composite(
    user_datasets=user_datasets,
    label_map=label_map,
    output_shape=output_shape,
    train_users=train_users,
    test_users=test_users,
    window_size=window_size,
    shift=window_size//2,
    normalise_dataset=True,
    verbose=1
)

(train_x_split, train_y_split) = np_train
(val_x_split, val_y_split) = np_val
(test_x, test_y_one_hot) = np_test

# Save the arrays
np.save(dataset_save_path + 'train_x_split.npy', train_x_split)
np.save(dataset_save_path + 'train_y_split.npy', train_y_split)
np.save(dataset_save_path + 'val_x_split.npy', val_x_split)
np.save(dataset_save_path + 'val_y_split.npy', val_y_split)
np.save(dataset_save_path + 'test_x.npy', test_x)
np.save(dataset_save_path + 'test_y_one_hot.npy', test_y_one_hot)'''

# Read the arrays
train_x_split = np.load(dataset_save_path + 'train_x_split.npy')
train_y_split = np.load(dataset_save_path + 'train_y_split.npy')
val_x_split = np.load(dataset_save_path + 'val_x_split.npy')
val_y_split = np.load(dataset_save_path + 'val_y_split.npy')
test_x = np.load(dataset_save_path + 'test_x.npy')
test_y_one_hot = np.load(dataset_save_path + 'test_y_one_hot.npy')

# Build three tuples
np_train = (train_x_split, train_y_split)
np_val = (val_x_split, val_y_split)
np_test = (test_x, test_y_one_hot)


# %%
batch_size = 512
decay_steps = 1000
epochs = 20
temperature = 0.1
transform_funcs = [
    # transformations.scaling_transform_vectorized, # Use Scaling trasnformation
    transformations.rotation_transform_vectorized # Use rotation trasnformation
]
transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)



# %%
start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = simclr_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()
print(simclr_model.input_shape)

with tf.device('/gpu:0'):
    trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

    simclr_model_save_path = f"{working_directory}{start_time_str}_simclr.hdf5"
    trained_simclr_model.save(simclr_model_save_path)

# %%
base_model.summary()

# %%
plt.figure(figsize=(5,2))
plt.plot(epoch_losses)
plt.ylabel("Loss", fontsize = 12)
# change font size of axis labels
plt.tick_params(axis='both', which='major', labelsize=6)
# change size of labels
plt.title("SimCLR Loss", fontsize = 12)

plt.xlabel("Epoch", fontsize = 12)
plt.show()

# %% [markdown]
# ## Fine-tuning and Evaluation

# %% [markdown]
# ### Linear Model

# %%
simclr_model_save_path


# %%
np_test[1][0] = [1., 0., 0., 0., 0., 0., 0.]

# %%
total_epochs = 10
batch_size = 200
tag = "linear_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
linear_evaluation_model = simclr_models.create_linear_model_from_base_model(simclr_model, output_shape, intermediate_layer=7)

linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

training_history = linear_evaluation_model.fit(
    x = np_train[0],
    y = np_train[1],
    batch_size=batch_size,
    shuffle=True,
    epochs=total_epochs,
    callbacks=[best_model_callback],
    validation_data=np_val
)

linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)

print("Model with lowest validation Loss:")
print(simclr_utitlities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Model in last epoch")
print(simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))


# %% [markdown]
# ### Full HAR Model

# %%

# %%
print(np_train[0].shape)
print(np_train[1].shape)
print(np_val[0].shape)
print(np_val[1].shape)
print(np_test[0].shape)
print(np_test[1].shape)

# %%
root = "/home/elian.riveros/dl-13-elian/notebooks/workspaces"
file_name = "lstm-har/gatednet_ax2Sns/dataset/UEA_archive/BasicMotions"
dataset = 'cola'
train_X_path = root + '/' + file_name + '/' + 'X_train_'+dataset+'_3ch.npy'
train_Y_path = root + '/' + file_name + '/' + 'y_train_'+dataset+'.npy'
val_X_path = root + '/' + file_name + '/' + 'X_val_'+dataset+'_3ch.npy'
val_Y_path = root + '/' + file_name + '/' + 'y_val_'+dataset+'.npy'        
test_X_path = root + '/' + file_name + '/' + 'X_test_'+dataset+'_3ch.npy'
test_Y_path = root + '/' + file_name + '/' + 'y_test_'+dataset+'.npy'

train_X = np.load(train_X_path)
train_Y = np.load(train_Y_path)
val_X = np.load(val_X_path)
val_Y = np.load(val_Y_path)
test_X = np.load(test_X_path)
test_Y = np.load(test_Y_path)

# %%
print("Shapes:")
print(train_X.shape, train_Y.shape)
print(val_X.shape, val_Y.shape)
print(test_X.shape, test_Y.shape)

# %%
train_Y_onehot= tf.one_hot(train_Y, 7)
val_Y_onehot= tf.one_hot(val_Y, 7)
test_Y_onehot= tf.one_hot(test_Y, 7)

# %%
type(np_train)

# %%
np_train2 = (train_X, train_Y_onehot)
np_val2 = (val_X, val_Y_onehot)

# %%
simclr_model_save_path = "test_run/20230614-021725_simclr.hdf5"

# %%
simclr_model_save_path

# %%
total_epochs = 50
batch_size = 200
tag = "full_eval"

simclr_model = tf.keras.models.load_model(simclr_model_save_path)
full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(simclr_model, output_shape, model_name="TPN", intermediate_layer=7, last_freeze_layer=4)

# %%
full_eval_best_model_file_name = f"{working_directory}{start_time_str}_simclr_{tag}.hdf5"
full_eval_best_model_file_name

# %%
best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

training_history = full_evaluation_model.fit(
    x = np_train2[0],
    y = np_train2[1],
    batch_size=batch_size,
    shuffle=True,
    epochs=total_epochs,
    callbacks=[best_model_callback],
    validation_data=np_val2
)

full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name)

print("Model with lowest validation Loss:")
print(simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Model in last epoch")
print(simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))


# %%
full_evaluation_model.summary()

# %% [markdown]
# ## Extra: t-SNE Plots

# %% [markdown]
# ### Parameters

# %%
# Select a model from which the intermediate representations are extracted
target_model = simclr_model
perplexity = 30.0


# %%
simclr_model.summary()

# %% [markdown]
# ### t-SNE Representations

# %%
intermediate_model = simclr_models.extract_intermediate_model_from_base_model(target_model, intermediate_layer=7)
intermediate_model.summary()

embeddings = intermediate_model.predict(np_test[0], batch_size=600)
tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=1, random_state=42)
tsne_projections = tsne_model.fit_transform(embeddings)



# %% [markdown]
# ### Plotting

# %%
labels_argmax = np.argmax(np_test[1], axis=1)
unique_labels = np.unique(labels_argmax)

plt.figure(figsize=(8,4))
graph = sns.scatterplot(
    x=tsne_projections[:,0], y=tsne_projections[:,1],
    hue=labels_argmax,
    palette=sns.color_palette("hsv", len(unique_labels)),
    s=50,
    alpha=1.0,
    rasterized=True
)
plt.xticks([], [])
plt.yticks([], [])


plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2)
legend = graph.legend_
for j, label in enumerate(unique_labels):
    legend.get_texts()[j].set_text(label_list_full_name[label])

# %% [markdown]
# ### Custom Color maps (Optional)
# 
# This section can be run to produce plots where semantically similar classes share similar colors. This requires the definition of a custom mapping of classes to colors.

# %%
# This is used to select colors for labels which are close to each other
# Each pair corresponds to one label class
# i.e. ['null', 'sitting', 'standing', 'walking', 'walking upstairs', 'walking downstairs', 'jogging']
# The first number determines the color map, and the second determines its value along the color map
# So 'sitting', 'standing' will share similar colors, and 'walking', 'walking upstairs', 'walking downstairs' will share another set of similar colors
label_color_spectrum = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0)]

# This step generates a list of colors for different categories of activities
# Here we assume 5 categories, and 5 different intesities within each category
major_colors = ['cool', 'Blues', 'Greens', 'Oranges', 'Purples']
color_map_base = dict (
    [((i, j), color) for i, major_color in enumerate(major_colors) for j, color in enumerate(reversed(sns.color_palette(major_color, 5))) ]
)
color_palette = np.array([color_map_base[color_index] for color_index in label_color_spectrum])

# %%
# This selects the appropriate number of colors to be used in the plot
labels_argmax = np.argmax(np_test[1], axis=1)
unique_labels = np.unique(labels_argmax)

plt.figure(figsize=(16,8))
graph = sns.scatterplot(
    x=tsne_projections[:,0], y=tsne_projections[:,1],
    hue=labels_argmax,
    palette=list(color_palette[unique_labels]),
    s=50,
    alpha=1.0,
    rasterized=True
)
plt.xticks([], [])
plt.yticks([], [])


plt.legend(loc='lower left', bbox_to_anchor=(0.25, -0.3), ncol=2)
legend = graph.legend_
for j, label in enumerate(unique_labels):
    legend.get_texts()[j].set_text(label_list_full_name[label])


# %%



