import pandas as pd
import xgboost as xgb
class sep_estimator:
    def __init__(self,params_0,params_1,params_2,params_3,cols0,cols1,cols2,cols3,
                 est0=xgb.XGBRegressor,est1=xgb.XGBRegressor,est2=xgb.XGBRegressor,est3=xgb.XGBRegressor):
        self.est0 = est0(**params_0)
        self.est1 = est1(**params_1)
        self.est2 = est2(**params_2)
        self.est3 = est3(**params_3)
        self.cols0 = cols0
        self.cols1 = cols1
        self.cols2 = cols2
        self.cols3 = cols3
    def preprocess_0(self,data,mode):
        if mode=='predict':
            assert (data['labels'].values==0).all()
        data = data.loc[:,self.cols0]
        return data
    def preprocess_1(self,data,mode):
        if mode=='predict':
            assert (data['labels'].values==1).all()
        data = data.loc[:,self.cols1]
        return data
    def preprocess_2(self,data,mode):
        if mode=='predict':
            assert (data['labels'].values==2).all()
        data = data.loc[:,self.cols2]
        return data
    def preprocess_3(self,data,mode):
        if mode=='predict':
            assert (data['labels'].values==3).all()
        data = data.loc[:,self.cols3]
        return data
    def fit(self,X,y):
        X0 = X.copy()
        X1 = X.copy()
        X2 = X.copy()
        X3 = X.copy()
        X0 = self.preprocess_0(X0,'train')
        X1 = self.preprocess_1(X1,'train')
        X2 = self.preprocess_2(X2,'train')
        X3 = self.preprocess_3(X3,'train')
        self.est0.fit(X0,y)
        self.est1.fit(X1,y)
        self.est2.fit(X2,y)
        self.est3.fit(X3,y)
    def predict(self,X):
        X0 = X.loc[X['labels']==0,:].copy()
        X1 = X.loc[X['labels']==1,:].copy()
        X2 = X.loc[X['labels']==2,:].copy()
        X3 = X.loc[X['labels']==3,:].copy()
        index_0 = X0.index
        index_1 = X1.index
        index_2 = X2.index
        index_3 = X3.index
        X0 = self.preprocess_0(X0,'predict')
        X1 = self.preprocess_1(X1,'predict')
        X2 = self.preprocess_2(X2,'predict')
        X3 = self.preprocess_3(X3,'predict')
        res = pd.DataFrame(index=X.index)
        if len(X0)>0:
            pred0 = self.est0.predict(X0)
            res.loc[index_0,0] = pred0
        if len(X1)>0:
            pred1 = self.est1.predict(X1)
            res.loc[index_1,0] = pred1
        if len(X2)>0:
            pred2 = self.est2.predict(X2)
            res.loc[index_2,0] = pred2
        if len(X3)>0:
            pred3 = self.est3.predict(X3)
            res.loc[index_3,0] = pred3
        return res[0].values.flatten()