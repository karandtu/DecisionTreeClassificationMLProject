from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

#generate the data using in-built skit-learn library called make_classification
def generate_synthetic_data_and_perform_tree_classification():

   x,y=make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,random_state=42)

#split the data into subsets xtrain xtest ytrain ytest and bring test results to a capacity
#size of 0.2 measure level within a random seed state.

   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train the DecisionTreeClassifier and perform prediction on new test-set
   treeClassifierVariable = DecisionTreeClassifier(x_train,y_train)
   treeClassifierVariable.fit(x_train,y_train)
   y_pred=treeClassifierVariable.predict(x_test)

   acc_score=accuracy_score(y_test,y_pred)
   cm=confusion_matrix(y_test,y_pred)
   print(f"Accuracy Score:",{acc_score})
   print(f"Confusion Matrix:",cm)




