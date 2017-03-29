import logging

import luigi
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from {{cookiecutter.project_name}}.inputs import KaggleInputFile
from {{cookiecutter.project_name}}.base import BaseTask


logger = logging.getLogger('luigi-interface')



class FitModel(BaseTask):

    def requires(self):
        yield KaggleInputFile()

    def grid_search(self, clf, data, y, cv, grid):
        start_time = datetime.datetime.now()
        gs = GridSearchCV(clf, grid, cv=cv, scoring='neg_log_loss')
        gs.fit(data, y)

        logger.info("""Time elapsed: {}\n
                       Best score: {}\n
                       Best estimator: {}\n
                    """.format(datetime.datetime.now() - start_time, gs.best_score_, gs.best_estimator_))
        return gs

    def run(self):
        df = pd.read_csv(self.input()[0].path)
        scaler = joblib.load(self.input()[1].path)
        df = df.dropna()
        cv = KFold(n_splits=5, shuffle=True, random_state=241)
        y = df[self.y].values
        df = df.drop(list(self.drop), axis=1)
        logger.info('Got {} games for fiting'.format(df.shape[0]))

        gs = self.fit(scaler.transform(df, y), y, cv)

        joblib.dump(gs.best_estimator_, self.output().path)


class LogReg(FitModel):

    def fit(self, df, y, cv):
        grid = {"logistic__C": 10. ** np.arange(-5, 5)}
        # create feature union
        features = []
        features.append(('pca', PCA(n_components=10)))
        features.append(('select_best', SelectKBest(k=20)))
        feature_union = FeatureUnion(features)
        # create pipeline
        estimators = []
        estimators.append(('feature_union', feature_union))
        estimators.append(('logistic', LogisticRegression(verbose=False)))
        model = Pipeline(estimators)

        gs = self.grid_search(model, df, y, cv, grid)

        self.plot_gs(gs, grid['logistic__C'], 'logistic__C.png')
        return gs

    def output(self):
        return luigi.LocalTarget(self.output_root + "/logistic.pkl")


if __name__ == "__main__":
    luigi.run()
