import logging

import luigi
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.externals import joblib
from {{cookiecutter.project_name}}.base import BaseTask
from {{cookiecutter.project_name}}.train import GetScaler, LogReg
from {{cookiecutter.project_name}}.inputs import KaggleInputData


logger = logging.getLogger('luigi-interface')

class Predict(BaseTask):

    y = luigi.Parameter(default='win')

    def requires(self):
        yield LogReg()
        yield GetScaler()

    def run(self):
        model = joblib.load(self.input()[0].path)
        scaler = joblib.load(self.input()[1].path)
        df = pd.read_csv(self.input()[2].path)
        df = df.drop(list(self.drop), axis=1)
        print(df.head())
        logger.info('Got {} games for prediction'.format(df.shape[0]))

        prediction = model.predict_proba(scaler.transform(df))[:, 1]
        prediction.to_csv(self.output().path, index=False)


class LogRegPred(Predict):

    def requires(self):
        yield RandomForest()
        yield GetScaler()

    def output(self):
        return luigi.LocalTarget(self.output_root + "/logreg_predictions.csv")


if __name__ == "__main__":
    luigi.run()
