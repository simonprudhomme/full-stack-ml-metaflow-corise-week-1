from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np

def labeling_function(row):
    label = np.nan
    if row['rating'] >= 4:
        label = 1
    elif row['rating'] < 4:
        label = 0
    else:
        label = np.nan
    return label

class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):
        # Start the Flow
        import io 
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({'label': labels, **_has_review_df})

        df = pd.DataFrame({'review': reviews, 'label': labels})
        X = df[['review']]
        y = df['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.split_size)
        print(f'num of rows in train set: {self.X_train.shape[0]}')
        print(f'num of rows in validation set: {self.X_test.shape[0]}')
        # ----> next 
        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute Baseline Model"
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.compose import make_column_transformer
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import accuracy_score, roc_auc_score

        # define pipeline
        #vectorizer = TfidfVectorizer()
        vectorizer = CountVectorizer()
        preprocessing_pipeline = make_column_transformer((vectorizer, 'review'), remainder='drop')
        model = DecisionTreeClassifier()
        self.pipeline = make_pipeline(preprocessing_pipeline, model) 
        
        # fit pipeline
        self.pipeline.fit(self.X_train, self.y_train)

        # evaluate pipeline
        self.y_pred = self.pipeline.predict(self.X_test)
        self.base_acc = accuracy_score(self.y_test, self.y_pred)
        self.base_rocauc = roc_auc_score(self.y_test, self.y_pred)
        # ----> next 
        self.next(self.end)

    @card(type='corise')
    @step
    def end(self):
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
            print(msg.format(round(self.base_acc,3), round(self.base_rocauc,3)))

            current.card.append(Markdown("# Womens Clothing Review Results"))
            current.card.append(Markdown("## Overall Accuracy"))
            current.card.append(Artifact(self.base_acc))

            self.cm = confusion_matrix(self.y_test, self.y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
            disp.plot()
            current.card.append(Image.from_matplotlib(disp.figure_, label='confusion_matrix'))
            
        except Exception as ex:
            print(ex)

if __name__ == '__main__':
    BaselineNLPFlow()
