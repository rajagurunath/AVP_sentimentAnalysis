import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate,KFold
from typing import List
from collections import Counter,defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import spacy

class SpacyModels(object):
    def __init__(self,path:str,model_filenames:List):
        self.path=path
        self.model_names_to_load=[spacy.load(name) for name in model_filenames]
    def spacy_classify(self,x,model):
        sent=model(x)
        return np.argmax(np.array(list(sent.cats.values())))
    def spacy_vector(self,x,model):
        sent=model(x)
        return sent.vector.reshape(1,-1)
    # def build_feat(self,df:pd.DataFrame,text_column:str):
    #     vector_combine=[]
    #     for model in self.model_names_to_load:
    #         vector=df[text_column].apply(lambda x:self.spacy_vector(x,model))
    #         vector_combine.append(vector)
    #     return np.hstact(vector_combine)

    def build_feat(self,df:pd.DataFrame,text_column:str):
        vector_combine=[]
        for model in self.model_names_to_load:
            vector=df[text_column].apply(lambda x:self.spacy_vector(x,model))
            print(vector.shape)
            if not vector[0].shape[1]==0:
                print(vector[0])
                vector_combine.append(vector)
        
        return np.array(vector_combine).reshape(-1,300)

    def enrich_linguistic_feat(self,spacy_model,df):
        # pos=[]
        # tag=[]
        # dep=[]
        # is_alpha=[]
        # is_stop=[]
        # for token in doc:
        #     pos.append(token.pos_)
        #     tag.append(token.tag_)
        #     dep.append(token.dep_)
        #     is_alpha.append(token.is_alpha)
        #     is_stop.append(token.is_stop)
        # d=defaultdict({})
        # d['pos']=Counter(pos)
        # d['tag']=Counter(tag)
        # d['dep']=Counter(dep)
        # d[]
        # pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        # # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        # #         token.shape_, token.is_alpha, token.is_stop)
        pos=df.text.apply(lambda x:Counter([token.pos_ \
             for token in spacy_model(x) if len(token.pos_)>0 ]))
        tag=df.text.apply(lambda x:Counter([token.tag_ \
             for token in spacy_model(x) if len(token.tag_)>0 ]))
        dep=df.text.apply(lambda x:Counter([token.dep_ \
             for token in spacy_model(x) if len(token.dep_)>0 ]))
        # is_alpha=train_data.text.apply(lambda x:Counter([token.pos_ \
        #      for token in spacy_model(x) if len(token.pos_)>0 ]))
        is_alpha=df.text.apply(lambda x:sum([token.is_alpha \
              for token in spacy_model(x)  ]))
        is_alpha.name="is_alpha"
        is_stop=df.text.apply(lambda x:sum([token.is_stop \
              for token in spacy_model(x)  ]))
        is_stop.name="is_stop"
        
        len_text=train_data.text.apply(lambda x:len(x))
        len_token=train_data.text.apply(lambda x:len(x.split(' ')))
        #sentcount=df.text.apply(lambda x:len([sent.text \
        #      for sent in spacy_model(x).sents  ]))
        #sentcount.name="sentcount"
        
        #ner=df.text.apply(lambda x:Counter([ent.label_ \
        #     for ent in spacy_model(x).ents if len(ent.label_)>0 ]))
        #ner.name="ner"
        posdf=pd.DataFrame.from_dict(pos.values.tolist()).fillna(0)
        tagdf=pd.DataFrame.from_dict(tag.values.tolist()).fillna(0)
        depdf=pd.DataFrame.from_dict(dep.values.tolist()).fillna(0)
        
        res=pd.concat([posdf,tagdf,depdf,is_alpha,is_stop,len_text,len_token],axis=1)
        print(res.head())
        return res

    def sentiment(self,df,model):
        spacy_sent=df.text.apply(lambda x :model(x).sentiment)
        textblobFeat=df.text.apply(lambda x:self.textblob_sentiment(x))
        
        textblobFeat=pd.DataFrame.from_dict(dict(textblobFeat)).T
        return pd.concat([textblobFeat,spacy_sent],axis=1)

    def textblob_sentiment(self,x):
        tmp=TextBlob(x)
        return (tmp.polarity,tmp.subjectivity)
    
    def return_spacy_features(self,df):
        df1=self.enrich_linguistic_feat(self.model_names_to_load[0],df)
        df2=self.sentiment(df,self.model_names_to_load[0])
        df3=pd.DataFrame(self.build_feat(df,'text'))
        return pd.concat([df1,df2,df3],axis=1)

class BagoFWords(TransformerMixin):
    def __init__(self,estimators:List,est_names:List):
        self.est_list=estimators
        self.est_names=est_names
        self.pipeline=Pipeline(list(zip(self.est_names,self.est_list)))    
    def make_features_train(self,df):
        return pd.DataFrame(self.pipeline.fit_transform(df.text.values),columns=['svd{}'.format(i) for i in range(5)])

    def make_features_test(self,df):
        return pd.DataFrame(self.pipeline.transform(df.text.values),columns=['svd{}'.format(i) for i in range(5)])
    



def modelBuilding(X,y,cat_features):

    X_train, X_validation, y_train, y_validation = train_test_split(X,\
         y, train_size=0.8, random_state=1234)
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.01,
        task_type = "GPU"
        #loss_function='CrossEntropy'
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_validation, y_validation),
        verbose=True
    )
    print('Model is fitted: ' + str(model.is_fitted()))
    print('Model params:')
    print(model.get_params())
    return model


if __name__=='__main__':
    train=pd.read_csv("train_F3WbcTw.csv")
    spacymodel=SpacyModels("./",["spacy_model_cnn","spacy_model_ensemble","spacy_model_bow"])
    #FeatDF1=spacymodel.return_spacy_features(train)
    #FeatDF1=pd.read_csv("feat1Train.csv").iloc[:,1:]
    #print("checking Feat 1",FeatDF1.shape)
    #FeatDF1.to_csv("feat1Train.csv")
    #wordFeat=BagoFWords([TfidfVectorizer(),TruncatedSVD(n_components=5)],['tfidf','svd'])
    #FeatDF2=wordFeat.make_features_train(train)
    #FeatDF2=pd.read_csv("feat2Train.csv").iloc[:,1:]
    #FeatDF2.to_csv("feat2Train.csv",index=False)
    #print("checking Feat 2",FeatDF2.shape)
    #feat=pd.concat([FeatDF1,FeatDF2,train.loc[:,'drug']],axis=1)

    #feat.to_csv("trainfeat.csv")
    feat=pd.read_csv("trainfeat.csv").iloc[:,1:]
    print("checking Total Feat ",feat.shape)
    model=modelBuilding(feat,train.loc[:,'sentiment'],cat_features=['drug'])

    model.save_model('catboost_model.bin')
    model.save_model('catboost_model.json', format='json')


    """
    For test
    """
    test=pd.read_csv("test.csv")
    testfeatdf1=pd.read_csv("testfeat1.csv").iloc[:,1:]
    #testfeatdf1=spacymodel.return_spacy_features(test)
    print(testfeatdf1.shape)

    #testfeatdf1.to_csv("testfeat1.csv")
    #testfeatdf2=wordFeat.make_features_test(test)
    #print(testfeatdf2.shape)
    testfeatdf2=pd.read_csv("testfeat2.csv").iloc[:,1:]
    #testfeatdf2.to_csv("testfeat2.csv")
    testfeat=pd.concat([testfeatdf1,testfeatdf2,test.loc[:,'drug']],axis=1)
    print(testfeat.shape)
    testfeat.to_csv("testfeat.csv")
    
    pred=model.predict(testfeat)
    
    print(pred[:5],pred.shape)
    pred=pd.DataFrame(pred)
    pred.index=test.unique_hash

    pred.to_csv("submission.csv")
    
