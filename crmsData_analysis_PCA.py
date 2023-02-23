import math
import statistics
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv("CRMS_Data4ML_VegIndex_V1.csv")

# add vegetation community type name
df['curVegIndex'].replace(1, 'Maidencane'  ,inplace=True)
df['curVegIndex'].replace(2, 'Three-square',inplace=True)
df['curVegIndex'].replace(3, 'Roseau'      ,inplace=True)
df['curVegIndex'].replace(4, 'Paspalum'    ,inplace=True)
df['curVegIndex'].replace(5, 'Wiregrass'   ,inplace=True)
df['curVegIndex'].replace(6, 'Bulltongue'  ,inplace=True)
df['curVegIndex'].replace(7, 'Needlerush'  ,inplace=True)
df['curVegIndex'].replace(8, 'Bulrush'     ,inplace=True)
df['curVegIndex'].replace(9, 'Brackish-Mix',inplace=True)
df['curVegIndex'].replace(10, 'Oyster'     ,inplace=True)
df['curVegIndex'].replace(11, 'Saltgrass'  ,inplace=True)

# # copy to vegetation index and % cover to veg_label and veg_perCover, respectively
# # veg_labels = df.curVegIndex
# # veg_perCover = df.curVegCover

# # convert series to array
# data_length = len(veg_labels)
# aa = pd.Series(veg_labels).array
# # reshape aa array and make ndarray
# bb = np.reshape(aa,(data_length,1))


# remove columns 'preVegIndex', 'preVegCover' from the dataset
veg_dataset = df.drop(['STID', 'Year', 'preVegIndex', 'preVegCover', 'curVegCover'], axis=1)

from sklearn.preprocessing import StandardScaler

features = ['avgSAL', 'minSAL', 'maxSAL', 'stdWL', 'avgWL2MA', 'avgTMP', 'minTMP', 'perFLD']
x = veg_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x)

# check dimension
# print(x.shape[1])
# Let's check whether the normalized data has a mean of zero and a standard deviation of one.
# print(np.mean(x),np.std(x))

# Let's convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_vegdata = pd.DataFrame(x,columns=feat_cols)

# Now comes the critical part, the next few lines of code will be projecting the thirty-dimensional Breast Cancer data to two-dimensional principal components.

# You will use the sklearn library to import the PCA module, and in the PCA method,
# you will pass the number of components (n_components=2) and finally call fit_transform on the aggregate data.
# Here, several components represent the lower dimension in which you will project your higher dimension data.

from sklearn.decomposition import PCA
pca_vegdata = PCA(n_components=3)
principalComponents_vegdata = pca_vegdata.fit_transform(x)

# Next, let's create a DataFrame that will have the principal component values for all 569 samples
principal_vegdata_Df = pd.DataFrame(data = principalComponents_vegdata, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
print(principal_vegdata_Df.tail())

# Once you have the principal components, you can find the explained_variance_ratio.
# It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
print('Explained variation per principal component: {}'.format(pca_vegdata.explained_variance_ratio_))

# Plot the visualization of the 4580 samples along the principal component - 1 and principal component - 2 axis.
# It should give you good insight into how your samples are distributed among the two classes.
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Vegetation Dataset",fontsize=20)

string = 'Explained variation per principal component: {}'.format(pca_vegdata.explained_variance_ratio_)
plt.text(-1, -4, string, fontsize=10)


targets =['Maidencane','Three-square','Roseau','Paspalum','Wiregrass','Bulltongue','Needlerush','Bulrush','Brackish-Mix','Oyster','Saltgrass']

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'b', 'g', 'r', 'c', 'm']
symbols = ['o', 'o', 'o', 'o', 'o', 'o', '^', '^', '^', '^', '^']

for target, color, symbol in zip(targets,colors,symbols):
    indicesToKeep = veg_dataset['curVegIndex'] == target
    plt.scatter(principal_vegdata_Df.loc[indicesToKeep, 'principal component 1']
               , principal_vegdata_Df.loc[indicesToKeep, 'principal component 2'], c = color, marker = symbol, s = 30)

plt.legend(targets,prop={'size': 10})


plt.show()
