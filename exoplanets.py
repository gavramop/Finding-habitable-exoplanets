# utilities
import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# nltk
import nltk
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy  as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import sklearn.cluster
import re

%load_ext autoreload
%autoreload 2


df = pd.read_csv('kepler_koi.csv',
                         comment='#',
                         infer_datetime_format=True)

df['kepler_name'] = df['kepler_name'].fillna('Unknown')
df['koi_prad'] = df['koi_prad'].fillna(0)
df['koi_insol'] = df['koi_insol'].fillna(0)

dataset = df[['kepler_name','koi_teq','koi_prad','koi_smass','koi_disposition','koi_insol']]


dataset['goldilocks_temp'] = ((273.2 <= dataset['koi_teq']) & (dataset['koi_teq'] <= 373.2))
goldilocks_counts = {}
goldilocks_counts['temp'] = {
    "too cold":   dataset.query('koi_teq         <= 273.2').shape[0],
    "just right": dataset.query('goldilocks_temp == 1' ).shape[0],
    "too hot":    dataset.query('koi_teq         >= 373.2').shape[0],
}
for key, value in goldilocks_counts['temp'].items():
    print( 'Exoplanets that are %-10s: %4d (%5.2f%%)' % ( key, value, 100*value/dataset.shape[0] ) )  


dataset['goldilocks_size'] = ((0.8 <= dataset['koi_prad']) & (dataset['koi_prad'] <= 1.7))
goldilocks_counts['size'] = {
    "too small":  dataset.query('koi_prad        <= 0.8' ).shape[0],
    "just right": dataset.query('goldilocks_size == True').shape[0],
    "too big":    dataset.query('koi_prad        >= 1.7' ).shape[0],
}
for key, value in goldilocks_counts['size'].items():
    print( 'Exoplanets that are %-10s: %4d (%5.2f%%)' % ( key, value, 100*value/dataset.shape[0] ) ) 


dataset['goldilocks'] = ((dataset['goldilocks_temp'] == True) & (dataset['goldilocks_size'] == True))
goldilocks_counts['combined'] = {
    "just right": dataset.query('goldilocks==True').shape[0]
}

for key in goldilocks_counts.keys():
    value = goldilocks_counts[key]['just right']
    print( 'Exoplanets that are "just right" %-10s: %4d (%5.2f%%)' % ( key, value, 100*value/dataset.shape[0] ) )  




sns.set(rc={'figure.figsize':(20,10)})
sns.scatterplot(
    data=dataset,        
    x="koi_teq",
    y="koi_prad",
    size="koi_prad", sizes=(20,400),    
    hue="goldilocks_temp", palette="RdBu",
)
plt.title('Confirmed Exoplanets in the Goldilocks Temperature')
plt.xlabel('Temperature (Kelvin)')
plt.ylabel('Radius (Earth Radii)')
display()

sns.set(rc={'figure.figsize':(20,10)})
sns.scatterplot(
    data=dataset.query('goldilocks_temp==True'),        
    x="koi_teq",
    y="koi_prad",
    size="koi_prad", sizes=(20,400),    
    hue="goldilocks", palette="RdBu",
)
plt.title('Confirmed Exoplanets with the Goldilocks Size and Temperature')
plt.xlabel('Temperature (Kelvin)')
plt.ylabel('Radius (Earth Radii)')
display()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(   "Number of potentually habitable exoplanets: " + str(dataset.query('goldilocks==True').shape[0]) )
    print(   "Names of potentually habitable exoplanets: " + ", ".join( dataset.query('goldilocks==True')['kepler_name'].tolist()) )    
    display( dataset.query('goldilocks==True') )


df = dataset.query('goldilocks==True')
df['KMeans_StarType'] = sklearn.cluster.KMeans(n_clusters=4).fit_predict(df[['koi_sma','koi_smass']])

plot = sns.scatterplot(
    data=df,        
    x="koi_sma",
    y="koi_smass",

    size="koi_prad", sizes=(20,400),    
    hue="KMeans_StarType", palette="Blues",
#     hue="koi_teq", palette="RdBu_r",
)
for line in range(0,df.shape[0]):
     plot.text(
         df['koi_sma'][line]+0.005, 
         df['koi_smass'][line], 
         df['kepler_name'][line], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         weight='semibold'
     )

plt.title('Confirmed Goldilocks Exoplanets by StarType')
plt.xlabel('Semi Major Axis / Orbital Distance (AU)')
plt.ylabel('Stellar Mass (solar mass)')
display()



df = dataset.query('goldilocks==True')
df['KMeans_PlanetType'] = sklearn.cluster.KMeans(n_clusters=6).fit_predict(df[['koi_smet','koi_prad']])

plot = sns.scatterplot(
    data=df,        
    x="koi_smet",
    y="koi_prad",

    size="koi_prad", sizes=(20,400),    
    hue="KMeans_PlanetType", palette="Accent",
)
for line in range(0,df.shape[0]):
     plot.text(
         df['koi_smet'][line]+0.005, 
         df['koi_prad'][line], 
         df['kepler_name'][line], 
         horizontalalignment='left', 
         size='medium', 
         color='black', 
         # weight='semibold'
     )

plt.title('Confirmed Goldilocks Exoplanets by PlanetType')
plt.xlabel('Stellar Metallicity')
plt.ylabel('Planetary Radius (Earth radii)')
display()


display(
    sns.scatterplot(
        data=dataset,
        x="ra", 
        y="dec",
        sizes=(200,20),
        size="goldilocks", 
        hue="goldilocks", palette="hot",
    )
)


dataset['goldilocks_temp'] = dataset.goldilocks_temp.astype(int)


#Separating positive and negative temp
data_pos = dataset[dataset['goldilocks_temp'] == 1]
data_neg = dataset[dataset['goldilocks_temp'] == 0]

#taking one fourth data so we can run on our machine easily
data_pos = data_pos.iloc[:int(400)]
data_neg = data_neg.iloc[:int(400)]

dataset = pd.concat([data_pos, data_neg])

sns.set(rc={'figure.figsize':(20,10)})
sns.scatterplot(
    data=dataset,        
    x="koi_teq",
    y="koi_prad",
    size="koi_prad", sizes=(20,400),    
    hue="goldilocks_temp", palette="RdBu",
)
plt.title('Confirmed Exoplanets in the Goldilocks Temperature')
plt.xlabel('Temperature (Kelvin)')
plt.ylabel('Radius (Earth Radii)')
display()


X = dataset[['koi_prad','koi_dor']]
Y = dataset.goldilocks_temp
# Separating the 90% data for training data and 10% for testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state =0)



# Model Logistic Regression
LRmodel = LogisticRegression()
LRmodel.fit(X_train, Y_train)
y_pred3 = LRmodel.predict(X_test)
print(classification_report(Y_test, y_pred3))

from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report

#ROC AUC curve
rocAuc = roc_auc_score(Y_test, y_pred3)

falsePositiveRate, truePositiveRate, _ = roc_curve(Y_test, y_pred3)

plt.figure()

plt.plot(falsePositiveRate, truePositiveRate, color='green',
         lw=3, label='ROC curve (area = %0.2f)' % rocAuc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of ideal Temperature')
plt.legend(loc="lower right")
plt.show()

#Other accuracy metrices
y_pred3 = (y_pred3 > 0.5)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_train,Y_train)
y_pred5 = LRmodel.predict(X_test)
print(classification_report(Y_test, y_pred5))


# Import various layers needed for the architecture from keras
import tensorflow
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
# The Input layer 
sequence_input = Input(shape=(4,), dtype='int32')
x = Dense(512, activation='relu')(sequence_input)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
# Passed on to activation layer to get final output
outputs = Dense(1, activation='sigmoid')(x)
model = tensorflow.keras.Model(sequence_input, outputs)


from keras.models import Model, Sequential

# define the keras model
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, Y, epochs=40, batch_size=8)


# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
