import pandas as pd
from sklearn.cluster import KMeans
class cluster_target_encoder:
    def __init__(self, nclusters = 4, seed=0):
        self.seed = seed
        self.nclusters = nclusters
    def make_encoding(self,df):
        self.encoding = df.groupby('X')['y'].mean()
    def fit(self,X,y):
        df = pd.DataFrame(columns=['X','y'],index=X.index)
        df['X'] = X
        df['y'] = y
        self.make_encoding(df)
        clust = KMeans(self.nclusters, random_state=self.seed)
        labels = clust.fit_predict(self.encoding[df['X'].values].values.reshape(-1,1))
        df['labels'] = labels
        self.clust_encoding = df.groupby('X')['labels'].median()
    def transform(self,X):
        res = X.map(self.clust_encoding).astype(float)
        return res
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)