import unittest
from src.misc import ETL, Analizer, Model, Retrain, CLIManager, ParameterControl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import unittest
from unittest.mock import patch, MagicMock
from main import ETL
import pandas as pd
import os

class TestETL(unittest.TestCase):

    def setUp(self):
        self.etl = ETL()

    def test_etl(self):
        self.etl.extract = MagicMock()
        self.etl.transform = MagicMock()
        self.etl.load = MagicMock(return_value=(pd.DataFrame(), pd.DataFrame()))

        self.etl.etl()
        self.etl.extract.assert_called_once()
        self.etl.transform.assert_called_once()
        self.etl.load.assert_called_once()

class TestAnalizer(unittest.TestCase):

    def setUp(self):
        self.analizer = Analizer()

    def test_analize(self):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        self.analizer._Analizer__select_features = MagicMock(return_value=(train_data, test_data))
        result = self.analizer.analize(train_data, test_data)
        self.analizer._Analizer__select_features.assert_called_once()
        self.assertEqual(result, (train_data, test_data))

class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pc = ParameterControl()
        self.df_train = pd.read_parquet(self.pc.get_path('titanic_modeling_train'))
        self.df_test = pd.read_parquet(self.pc.get_path('titanic_modeling_test'))

    def setUp(self):
        self.model = Model()
        self.model.load_data(self.df_train, self.df_test)

    def test_load_data(self):
        self.assertIsNotNone(self.model.df_train)
        self.assertIsNotNone(self.model.df_test)
        self.assertIsNotNone(self.model.X_train)
        self.assertIsNotNone(self.model.X_val)
        self.assertIsNotNone(self.model.y_train)
        self.assertIsNotNone(self.model.y_val)
        self.assertEqual(len(self.model.X_train) + len(self.model.X_val), len(self.model.df_train))

    @patch('src.misc.Model.logistic_regression')
    @patch('src.misc.Model.random_forest')
    @patch('src.misc.Model.svm')
    @patch('src.misc.Model.xgboost')
    @patch('src.misc.Model.knn')
    def test_train(self, mock_knn, mock_xgboost, mock_svm, mock_random_forest, mock_logistic_regression):
        retrain = True
        self.model.train(retrain=retrain)
        mock_logistic_regression.assert_called_once_with(retrain=retrain)
        mock_random_forest.assert_called_once_with(retrain=retrain)
        mock_svm.assert_called_once_with(retrain=retrain)
        mock_xgboost.assert_called_once_with(retrain=retrain)
        mock_knn.assert_called_once_with(retrain=retrain)

    def test_validate(self):
        self.model.models = {
            'Logistic Regression': LogisticRegression()
        }
        self.model.models['Logistic Regression'].fit(self.model.X_train, self.model.y_train)
        self.model.validate()
        self.assertIn('Logistic Regression', self.model.model_scores)
        self.assertIn('accuracy', self.model.model_scores['Logistic Regression'])
        self.assertIn('f1', self.model.model_scores['Logistic Regression'])
        self.assertIn('roc_auc', self.model.model_scores['Logistic Regression'])

    @patch('src.misc.register_model')
    def test_registry_models(self, mock_register_model):
        self.model.models = {
            'Logistic Regression': LogisticRegression()
        }
        self.model.searches = {
            'Logistic Regression': MagicMock(best_params_={'model__param': 'value'})
        }
        self.model.model_scores = {
            'Logistic Regression': {'accuracy': 0.9, 'f1': 0.8, 'roc_auc': 0.85}
        }
        self.model.registry_models()
        mock_register_model.assert_called_once()


    @patch('src.misc.transition_model_to_production')
    def test_publish(self, mock_transition_model_to_production):
        self.model.model_scores = {
            'Logistic Regression': {'accuracy': 0.9, 'f1': 0.8, 'roc_auc': 0.85}
        }
        self.model.models = {
            'Logistic Regression': LogisticRegression()
        }
        self.model.publish(local=True)
        mock_transition_model_to_production.assert_called_once_with("Titanic Model - Logistic Regression", 1)


if __name__ == '__main__':
    unittest.main()
