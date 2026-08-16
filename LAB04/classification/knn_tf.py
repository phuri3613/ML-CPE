from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_knn(X_train, y_train, X_test, y_test, k_values):
    results = {}
    models = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[k] = acc
        models[k] = (knn, y_pred)
        
    return results, models