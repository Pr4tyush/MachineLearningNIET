
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[2]:

iris = load_iris()


# In[3]:

type(iris)


# In[4]:

iris.data


# In[5]:

print(iris.feature_names)


# In[6]:

print(iris.target)


# In[7]:

print(iris.target_names)


# In[8]:

plt.scatter(iris.data[:,1],iris.data[:,2],c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.show()

plt.scatter(iris.data[:,0],iris.data[:,3],c=iris.target, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[3])
plt.show()


# In[9]:

p = iris.data
q = iris.target


# In[10]:

print(p.shape)
print(q.shape)


# In[11]:

from sklearn.cross_validation import train_test_split
p_train,p_test,q_train,q_test = train_test_split(p,q,test_size=0.2,random_state=4)


# In[12]:

print(p_train.shape)
print(p_test.shape)


# In[13]:

print(q_train.shape)
print(q_test.shape)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(p_train,q_train)
        q_pred=knn.predict(p_test)
        scores[k] = metrics.accuracy_score(q_test,q_pred)
        scores_list.append(metrics.accuracy_score(q_test,q_pred))


# In[15]:

scores


# In[16]:

#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[17]:

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(p,q)


# In[19]:

#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'setosa',1:'versicolor',2:'virginica'}


x_new = [[3,4,5,2],
         [5,4,2,2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])


# In[ ]:



