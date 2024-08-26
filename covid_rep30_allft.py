#!/usr/bin/env python
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None

import pandas as pd
import numpy as np
import os
import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, DeepGraphInfomax

from tensorflow.keras import layers, optimizers, losses, metrics, Model, initializers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from statistics import mean
from statistics import stdev
import pickle
import sys

stdoutOrigin=sys.stdout
sys.stdout = open("covid_rep30_allft.txt", "w")

seed_value= 1234
#from sacred import Experiment
#ex = Experiment()
# Report it that number to your experiment tracking system.
#experiment = Experiment(project_name="Classification model")
#experiment.log_other("random seed", seed_value)
# 1. Carefully set that seed variable for all of your frameworks:
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)




feature_names = [f"w{i}" for i in range(2)]

raw_content = pd.read_csv(
    "contact_train_ft.csv",
    sep=",",  # tab-separated
)

content_str_subject = raw_content.set_index("id")
subject = content_str_subject["test"]
node_subjects=subject
node_subjects.value_counts().to_frame()   ########## testing results of the node with labels

feature_names = [f"w{i}" for i in range(2)]

full_content = pd.read_csv(
    "node_new_ft.csv",
    sep=",",  # tab-separated
)


full_str_subject = full_content.set_index("id")
full_str_subject
content_no_subject = full_str_subject.drop(columns="test")
content_no_subject

edges = pd.read_csv(
    "edge.csv",
    sep=",",  # tab-separated
)


from stellargraph import StellarGraph
no_subject = StellarGraph({"info": content_no_subject}, {"status": edges})
print(no_subject.info())
G=no_subject  ########## no response information; contain seed and contact 


gcn_auc_lis=[]
lr_auc_lis=[]
rf_auc_lis=[]

import random
import numpy as np

for i in range(30):

    random.seed(seed_value+i)
    ##### Random seed given
    np.random.seed(seed_value+i)

    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=5320, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=1773, test_size=None, stratify=test_subjects
    )
    df = pd.DataFrame({"Label": test_subjects})
    df.to_csv('test_label.csv')


    ### Converting to numeric arrays
    train_targets = np.array(pd.get_dummies(pd.DataFrame(train_subjects), columns=["test"]))
    val_targets = np.array(pd.get_dummies(pd.DataFrame(val_subjects), columns=["test"]))
    test_targets = np.array(pd.get_dummies(pd.DataFrame(test_subjects), columns=["test"]))


    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'), 
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
        metrics.AUC(name='prc', curve='PR') # precision-recall curve
    ]



    ### Creating the GCN layers
    generator = FullBatchNodeGenerator(G, method="gcn")

    train_gen = generator.flow(train_subjects.index, train_targets[:,1])
    gcn = GCN(
        layer_sizes=[16,16], activations=["elu","elu"], generator=generator, dropout=0.4,kernel_regularizer=regularizers.l2(5e-4),
    )
    x_inp, x_out = gcn.in_out_tensors()

    outputs = layers.Dense(units=1, activation="sigmoid")(x_out)

    model = Model(inputs=x_inp, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss=losses.binary_crossentropy,
        metrics=metrics.AUC(name='auc'),
    )
    val_gen = generator.flow(val_subjects.index, val_targets[:,1])

    es_callback = EarlyStopping(monitor="val_auc", patience=500, mode='max', restore_best_weights=True)


    history = model.fit(
        train_gen,
        epochs=1000,
        validation_data=val_gen,
        verbose=1,
        shuffle=False,  
        callbacks=[es_callback],)


    import matplotlib.pyplot as plt
    test_nodes = test_subjects.index    ############## here the node_subjects only contain the training data, contacts
    test_gen = generator.flow(test_nodes)
    test_predictions = model.predict(test_gen, test_targets)

    ########## nowe get auc , and other metrics using test_predictions, and test_subjects
    from sklearn import preprocessing

    test_lb = preprocessing.LabelBinarizer()
    test_lb=np.array(test_lb.fit_transform(np.array(test_subjects)))
    testy = test_lb.reshape([test_lb.shape[0],])
    gcn_probs = test_predictions[0,:,0]  
    gcn_probs=gcn_probs.reshape([1775,1])


    import matplotlib as mpl
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score


    gcn_auc = roc_auc_score(testy, gcn_probs)

    gcn_auc_lis.append(gcn_auc)




with open("gcn_auc_lis",'wb') as f:
    pickle.dump(gcn_auc_lis, f)

 
print(mean(gcn_auc_lis),
stdev(gcn_auc_lis)
)


sys.stdout.close()
sys.stdout=stdoutOrigin