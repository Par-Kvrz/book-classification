# In[2]:
import nltk
import sklearn
from sklearn import metrics
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
import re
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# In[3]:
text_1 = gutenberg.open('10985.txt').read() #The Infant System by Samuel Wilderspin
text_2 = gutenberg.open('42547.txt').read() #The Art and Practice of Silver Printing by Abney and Robinson
text_3 = gutenberg.open('10773.txt').read() #Ancient and Modern Physics by Thomas Edgar Willson
text_4 = gutenberg.open('51397.txt').read() #People Soup by Alan Arkin
text_5 = gutenberg.open('17699.txt').read() #The Evolution of Love by Emil Lucka
text_6 = gutenberg.open('29420.txt').read() #American Rural Highways by T. R. Agg
text_7 = gutenberg.open('389.txt').read()   #The Great God Pan by Arthur Machen

# In[18]:
class Preprocess(): #this class is for preprocessing and labeling the documents
    def __init__(self, text_1 , text_2 , text_3 , text_4 , text_5 , text_6 , text_7 ):
        self.text_1 = text_1
        self.text_2 = text_2
        self.text_3 = text_3
        self.text_4 = text_4
        self.text_5 = text_5
        self.text_6 = text_6
        self.text_7 = text_7
        self.raw_text = [self.text_1, self.text_2, self.text_3 , self.text_4 , self.text_5 , self.text_6, self.text_7 ]
        self.stopword = set(stopwords.words('english'))

    def cleaning(self, raw_text):
        clean_text = []
        for i in range(len(raw_text)):
             clean_text.append(re.sub('[^a-zA-Z]',' ', raw_text[i] ))
        return clean_text
        
    def tokenizing(self, clean_text):
        tokenized_word = []
        for g in range(len(clean_text)):
            tokenized_word.append(nltk.word_tokenize(clean_text[g]))
        return tokenized_word
    
    def stop_word(self, tokenized_word):
        tokenized_word_s = [word for word in tokenized_word if not word in self.stopword]
        return tokenized_word_s
    
    def randomizing(self, text):
        random_text = []
        for c in range(len(text)):
            for o in range(200):
                random.seed(o)
                random_text.append(random.sample(text[c], 150))
        return random_text
    
    def get_label(self, random_text):
        df = []        
        for b in range(len(random_text)):
            if b <= 200:
                df1 = pd.DataFrame()
                df1['text']  = [random_text[b].lower()]
                df1['authors'] = 'a'            
                df.append(df1)
            elif b>200 and b<=400:
                df2 = pd.DataFrame()
                df2['text']  = [random_text[b].lower()]
                df2['authors'] = 'b'
                df.append(df2)
            elif b>400 and b<=600:
                df3 = pd.DataFrame()
                df3['text']  = [random_text[b].lower()]
                df3['authors'] = 'c'
                df.append(df3)
            elif b>600 and b<=800:
                df4 = pd.DataFrame()
                df4['text']  = [random_text[b].lower()]
                df4['authors'] = 'd'
                df.append(df4)
            elif b>800 and b<=1000:
                df5 = pd.DataFrame()
                df5['text']  = [random_text[b].lower()]
                df5['authors'] = 'e'
                df.append(df5)
            elif b>1000 and b<=1200:
                df6 = pd.DataFrame()
                df6['text']  = [random_text[b].lower()]
                df6['authors'] = 'f'
                df.append(df6)
            elif b>1200 and b<=1400:
                df7 = pd.DataFrame()
                df7['text']  = [random_text[b].lower()]
                df7['authors'] = 'g'
                df.append(df7)
        return df
    
    
    def preprocess(self, raw_text):
        clean_text = self.cleaning(raw_text)
        tokenized_word = self.tokenizing(clean_text)
        tokenized_word_lower = []
        tokenized_word_lower_a =[]
        for dc in range(len(tokenized_word)):
            for ds in range(len(tokenized_word[dc])):
                tokenized_word_lower.append(tokenized_word[dc][ds].lower())
            tokenized_word_lower_a.append(tokenized_word_lower)
        tokenized_word_s= []
        for cd in range(len(tokenized_word_lower_a)):
            tokenized_word_s.append(self.stop_word(tokenized_word[cd]))
        random_text = self.randomizing(tokenized_word_s)
        random_text_plus= []
        for g in range(len(random_text)):
            random_text_plus.append(' '. join(random_text[g]))            
        df_corpus = self.get_label(random_text_plus)
        df_corpus = pd.concat([df_corpus[v] for v in range(len(df_corpus))], ignore_index = True)
        corpus = []
        for k in range(len(df_corpus)):
            corpus.append(df_corpus['text'][k].lower())       
        return corpus, df_corpus
        
# In[19]:
class Transform(): #this class is for transorming the documents to bow , tfidf and also applying the n_grams
    def __init__(self, pp_corpus, pp_df_corpus, n_gram = 1):
        self.vectorize_bow = CountVectorizer(ngram_range = (n_gram, n_gram))
        self.encoder = LabelEncoder()
        self.vectorize_tfidf = TfidfVectorizer(ngram_range = (n_gram, n_gram))
        self.pp_text = pp_corpus
        self.corp = pp_df_corpus
        self.tt_split = train_test_split
        self.n_gram = n_gram
    
    def vectorizing_bow(self):
        
        x_vector_bow = self.vectorize_bow.fit_transform(self.pp_text).toarray()
        y_vector_bow = self.corp.iloc[:, 1]
        x_bow = pd.DataFrame(x_vector_bow)
        y_vector_encoded = self.encoder.fit_transform(y_vector_bow)
        y_vector_encoded = pd.DataFrame(y_vector_encoded)
        return x_bow , y_vector_encoded
        
    def vectorizing_tfidf(self):
        x_vector_tfidf =  self.vectorize_tfidf.fit_transform(self.pp_text).toarray()
        x_tfidf = pd.DataFrame(x_vector_tfidf)
        return x_tfidf
    
    def training_test_set(self):
        x_bow, y_vector_encoded = self.vectorizing_bow()
        x_tfidf = self.vectorizing_tfidf()    
        x_train_bow, x_test_bow, y_train_bow, y_test_bow = self.tt_split(x_bow, y_vector_encoded , test_size = 0.30, random_state = 100)
        y_train_bow = pd.Series(y_train_bow.iloc[:,0])
        y_test_bow = pd.Series(y_test_bow.iloc[:,0])
        x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = self.tt_split(x_tfidf, y_vector_encoded , test_size = 0.30, random_state = 100)
        y_train_tfidf = pd.Series(y_train_tfidf.iloc[:,0])
        y_test_tfidf = pd.Series(y_test_tfidf.iloc[:,0])
        return  x_train_bow, x_test_bow, y_train_bow, y_test_bow, x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf


# In[20]:
class Train(): #this class is for training the data with svm,knn and decision tree
    def __init__(self, x_train_bow, x_test_bow, y_train_bow, y_test_bow, x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf ):
        self.x_train_bow = x_train_bow
        self.x_test_bow = x_test_bow
        self.y_train_bow = y_train_bow
        self.y_train_bow = y_test_bow
        self.x_train_tfidf = x_train_tfidf
        self.x_test_tfidf = x_test_tfidf
        self.y_train_tfidf = y_train_tfidf
        self.y_test_tfidf = y_test_tfidf
        self.clf_dt_bow = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        self.clf_dt_tfidf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        self.clf_svm_bow = SVC(kernel = 'rbf', random_state = 0, C=1, gamma = 0.01)
        self.clf_svm_tfidf = SVC(kernel = 'linear', random_state = 0, C=1)
        self.clf_KNN_bow = KNeighborsClassifier(n_neighbors = 5)
        self.clf_KNN_tfidf = KNeighborsClassifier(n_neighbors = 5)
        
    def svm(self):        
        self.clf_svm_bow.fit(x_train_bow, y_train_bow)
        y_pred_svm_bow = self.clf_svm_bow.predict(x_test_bow)
        self.clf_svm_tfidf.fit(x_train_tfidf, y_train_tfidf)
        y_pred_svm_tfidf = self.clf_svm_tfidf.predict(x_test_tfidf)
        return y_pred_svm_bow, y_pred_svm_tfidf 
    
    def decision_tree(self):
        self.clf_dt_bow.fit(x_train_bow, y_train_bow)
        y_pred_dt_bow = self.clf_dt_bow.predict(x_test_bow)
        self.clf_dt_tfidf.fit(x_train_tfidf, y_train_tfidf)
        y_pred_dt_tfidf = self.clf_dt_tfidf.predict(x_test_tfidf)
        return y_pred_dt_bow, y_pred_dt_tfidf
        
    def knn(self):
        self.clf_KNN_bow.fit(x_train_bow, y_train_bow)
        y_pred_KNN_bow = self.clf_KNN_bow.predict(x_test_bow)
        self.clf_KNN_tfidf.fit(x_train_tfidf, y_train_tfidf)
        y_pred_KNN_tfidf = self.clf_KNN_tfidf.predict(x_test_tfidf)
        return y_pred_KNN_bow, y_pred_KNN_tfidf 
        
    def ten_fold(self, name_of_the_estimator, x_train, y_train):
        accuracies = cross_val_score(estimator = name_of_the_estimator , X = x_train, y = y_train, cv = 10, n_jobs = -1)
        accu_avg = accuracies.mean()
        return accuracies , accu_avg

# In[21]:
class Erorr_Analysis(): #for finding the errors
    def __init__(self):
        self.y_test_bow = y_test_bow
        self.y_test_tfidf= y_test_tfidf
        self.y_pred_KNN_bow = y_pred_KNN_bow  
        self.y_pred_KNN_tfidf = y_pred_KNN_tfidf
        self.y_pred_dt_bow = y_pred_dt_bow, 
        self.y_pred_dt_tfidf = y_pred_dt_tfidf
        self.y_pred_svm_bow = y_pred_svm_bow
        self.y_pred_svm_tfidf = y_pred_svm_tfidf
        self.cm_knn_bow = confusion_matrix(y_test_bow, self.y_pred_KNN_bow)
        self.cm_knn_tfidf = confusion_matrix(y_test_tfidf, y_pred_KNN_tfidf)
        self.cm_dt_bow = confusion_matrix(y_test_bow, y_pred_dt_bow)
        self.cm_dt_tfidf = confusion_matrix(y_test_tfidf, y_pred_dt_tfidf) 
        self.cm_svm_bow = confusion_matrix(y_test_bow, y_pred_svm_bow)
        self.cm_svm_tfidf = confusion_matrix(y_test_tfidf, y_pred_svm_tfidf)
        
    
    def confusion_matrix_report(self, y_predicted_bow, y_predicted_tfidf):
        matrix_bow = (metrics.classification_report(y_test_bow, y_predicted_bow))
        matrix_tfidf = (metrics.classification_report(y_test_tfidf, y_predicted_tfidf))
        return matrix_bow, matrix_tfidf
    
    def misclassified(self, y_test, y_predicted, confusion_matrix):
        misclasses=[] 
        mislocs=[]
        i = 0
        for test in y_test:
            if test != y_predicted[i] and confusion_matrix[test,y_predicted[i]]>=1:
                misclasses.append(f"{test} was predicted wrongly as {y_predicted[i]}")
                mislocs.append(y_test.index[i])               
            i+=1
        misclassification = pd.DataFrame({'doc_number' : mislocs, 'misclassifiation' : misclasses} )
        return mislocs, misclassification
    
    def find_similar_errors(self, x_test, misloc ): #finding the word in other documents in the test-set
        row_len = int(x_test.size/len(x_test))
        error_doc= []
        test_set= []
        all_test= []
        errors= []
        find_similar_words=[]
        for o in range(len(mislocs)):
            for index, row in x_test.iterrows():
                if index == mislocs[o]:
                    for k in range(0,row_len):
                        error_doc.append(row[k])
        
                        
        for index, row in x_test.iterrows():
            for k in range(0,row_len):
                all_test.append(row[k])
                
        while all_test != []:
            test_set.append(all_test[:row_len])
            all_test = all_test[row_len:]
        while error_doc != []:
            errors.append(error_doc[:row_len])
            error_doc = error_doc[row_len:]
                
        for n in errors:    
            for index in test_set:
                for z in range(row_len):
                    if n[z] > 0 and index[z]> 0 :
                        find_similar_words.append(z)
                                        
        return find_similar_words
    
    def get_features(self, x_test, misloc):      
        get_words = trf.vectorize_bow
        feature = get_words.get_feature_names()
        similar = self.find_similar_errors(x_test, misloc)
        repeating_words=[]
        repeating_errors = []
        for f in range(100):         #for reducing the computation time
            random.seed(2)
            random_n = np.random.randint(0 , len(similar))
            repeating_errors.append(similar[random_n])
        for index in repeating_errors:
            repeating_words.append(feature[index])    
        corpus_repeat = [" ".join(str(word) for word in repeating_words)]
        return corpus_repeat, repeating_words, feature
    
    def error_result(self, corpus_repeat):
        from sklearn.feature_extraction.text import CountVectorizer
        count_vectorizer = CountVectorizer()
        error_repeat = count_vectorizer.fit_transform(corpus_repeat).toarray()
        error_words = pd.Series(count_vectorizer.get_feature_names())
        error_repeat = pd.Series(error_repeat.T[:,0])
        freq_word = pd.DataFrame({'word' : error_words , 'frequency' : error_repeat})
        return freq_word, error_words

# In[22]:
class Visualization():
    def __init__(self, corpus_repeat, repeating_words, error_words):
        self.corpus_repeat = corpus_repeat
        self.repeating_words = repeating_words
        self.error_words = error_words
        
    def word_cloud(self):
        w_cloud = WordCloud(max_font_size=70).generate(self.corpus_repeat[0])
        plt.figure(figsize=(16,12))
        plt.imshow(w_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    def word_freq(self):
        freqdist = nltk.FreqDist(self.repeating_words)
        plt.figure(figsize=(20,5))
        freqdist.plot(100)
        
    def ten_fold_v(self, accuracy_estimator):       
        plt.figure()
        plt.plot(np.array([0,1,2,3,4,5,6,7,8,9]),accuracy_estimator, 'r-', marker = 'o')
        plt.xlabel('ten fold')
        plt.ylabel('accuracy')
        plt.title('Ten fold cross validation')
        

# In[23]:
prp = Preprocess( text_1 , text_2 , text_3 , text_4 , text_5 , text_6 , text_7 )
pp_corpus, pp_df_corpus = prp.preprocess(prp.raw_text)

# In[25]:
trf = Transform(pp_corpus, pp_df_corpus)
x_train_bow, x_test_bow, y_train_bow, y_test_bow, x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = trf.training_test_set()

# In[26]:
train = Train( x_train_bow, x_test_bow, y_train_bow, y_test_bow, x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf)
y_pred_KNN_bow, y_pred_KNN_tfidf = train.knn()
y_pred_svm_bow, y_pred_svm_tfidf = train.svm()
y_pred_dt_bow, y_pred_dt_tfidf = train.decision_tree()

# In[32]:
eris = Erorr_Analysis()
mislocs , misclassification = eris.misclassified(eris.y_test_bow, y_pred_KNN_bow, eris.cm_knn_bow)
corpus_repeat, repeating_words , feature = eris.get_features(x_test_bow, mislocs)
freq_word, error_words = eris.error_result(corpus_repeat)

# In[33]:
misclassification

# In[34]:
viz = Visualization(corpus_repeat, repeating_words, error_words)

# In[36]:
viz.word_cloud()

# In[40]:
viz.ten_fold_v(accum)

# In[41]:
accum , accu = train.ten_fold(train.clf_KNN_bow, x_train_bow, y_train_tfidf)

# In[44]:
matrix_bow, matrix_tfidf = eris.confusion_matrix_report(y_pred_svm_bow, y_pred_svm_tfidf)

# In[46]:
print(matrix_tfidf)