from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



def train_knn_classifier(train_df, test_df, save_results = True):

    x_train = train_df.drop[]s