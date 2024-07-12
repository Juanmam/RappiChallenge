import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.impute import KNNImputer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import f_oneway
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import subprocess
import zipfile
from typing import Callable, Dict, List, Tuple, Union, Any
import inspect
import json
from contextlib import contextmanager
import random

class Retrain:
    """Class to represent the retrain flag."""
    value = False

class CLIManager:
    """
    A class to manage command-line interface commands and their execution.
    """

    def __init__(self):
        """Initialize the CLIManager with an empty command registry."""
        self.commands: Dict[str, Callable[..., Any]] = {}
        self.outputs: Dict[str, Any] = {}

    def register_command(self, command: str, function: Callable[..., Any]) -> None:
        """
        Register a command with its corresponding function.

        :param command: The command string to register.
        :type command: str
        :param function: The function to call when the command is executed.
        :type function: Callable[..., Any]
        """
        self.commands[command] = function

    def execute_command(self, command: str, **kwargs: Any) -> Any:
        """
        Execute a registered command with the provided keyword arguments.

        :param command: The command to execute.
        :type command: str
        :param kwargs: Keyword arguments to pass to the command's function.
        :type kwargs: Any
        :return: The result of the command execution.
        :rtype: Any
        :raises ValueError: If the command is not registered.
        """
        if command not in self.commands:
            raise ValueError(f"Command '{command}' not recognized.")
        
        result = self.commands[command](**kwargs)
        self.outputs[command] = result
        return result

    def run_interactive_mode(self) -> None:
        """Run the CLIManager in interactive mode."""
        print("Entering interactive mode. Type 'exit' to quit, 'help' for list of commands.")
        while True:
            user_input = input("> ")
            if user_input.strip().lower() == "exit":
                break
            if user_input.strip().lower() == "help":
                self.list_commands()
                continue

            try:
                result = self.execute_piped_commands(user_input)
                if result is not None:
                    print(f"The result is: {result}")
            except Exception as e:
                print(f"Error: {e}")

    def parse_arguments(self, command: str, args: List[str], input_data: Any = None) -> Dict[str, Any]:
        """
        Parse command line arguments for a specific command, handling both positional and keyword arguments.

        :param command: The command for which arguments are being parsed.
        :type command: str
        :param args: List of arguments.
        :type args: List[str]
        :param input_data: Optional input data from previous command in the pipe.
        :type input_data: Any
        :return: Dictionary of parsed arguments.
        :rtype: Dict[str, Any]
        """
        if command not in self.commands:
            raise ValueError(f"Command '{command}' not recognized.")
        
        func = self.commands[command]
        sig = inspect.signature(func)
        kwargs = {}
        positional_args = []
        i = 0

        while i < len(args):
            if args[i].startswith("--"):
                key = args[i][2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    kwargs[key] = args[i + 1]
                    i += 2
                else:
                    kwargs[key] = True
                    i += 1
            else:
                positional_args.append(args[i])
                i += 1

        param_iter = iter(sig.parameters.values())
        if input_data is not None:
            kwargs[next(param_iter).name] = input_data

        for pos_arg in positional_args:
            if pos_arg.endswith('.output'):
                referenced_command = pos_arg[:-7]
                if referenced_command in self.outputs:
                    output = self.outputs[referenced_command]
                    if isinstance(output, tuple):
                        for out_value in output:
                            param = next(param_iter)
                            kwargs[param.name] = out_value
                    else:
                        param = next(param_iter)
                        kwargs[param.name] = output
                else:
                    raise ValueError(f"Referenced command '{referenced_command}' not found.")
            else:
                param = next(param_iter)
                kwargs[param.name] = pos_arg

        for key, value in kwargs.items():
            if key in sig.parameters:
                param = sig.parameters[key]
                if param.annotation != inspect.Parameter.empty:
                    try:
                        kwargs[key] = param.annotation(value)
                    except ValueError as ve:
                        raise ValueError(f"Could not convert argument '{key}' to type {param.annotation}: {ve}")

        return kwargs

    def list_commands(self) -> None:
        """List all registered commands."""
        print("Available commands:")
        for command in self.commands:
            print(f"  {command}")

    def show_help(self, command: str) -> None:
        """
        Show help information for a specific command.

        :param command: The command to show help for.
        :type command: str
        """
        if command not in self.commands:
            print(f"Command '{command}' not recognized.")
            return
        
        func = self.commands[command]
        doc = inspect.getdoc(func)
        print(f"Help for command '{command}':")
        if doc:
            print(doc)
        else:
            print("No documentation available.")

    def run(self) -> None:
        """Run the CLIManager, processing arguments from the command line."""
        if len(sys.argv) < 2:
            self.run_interactive_mode()
            return

        user_input = " ".join(sys.argv[1:])

        try:
            result = self.execute_piped_commands(user_input)
            if result is not None:
                print(f"The result is: {result}")
        except Exception as e:
            print(f"Error: {e}")

    def execute_piped_commands(self, command_string: str) -> Any:
        """
        Execute a series of piped commands.

        :param command_string: The entire command input string.
        :type command_string: str
        :return: The final result after executing all piped commands.
        :rtype: Any
        """
        commands = command_string.split('|')
        input_data = None

        for command in commands:
            parts = command.strip().split()
            cmd_name = parts[0]
            args = parts[1:]

            kwargs = self.parse_arguments(cmd_name, args, input_data)
            input_data = self.execute_command(cmd_name, **kwargs)

        return input_data

class Singleton(type):
    """A Singleton metaclass to ensure only one instance of the ParameterControl class is created."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ParameterControl(metaclass=Singleton):
    """A class to control parameters loaded from a JSON file."""

    def __init__(self, file_path: str = "parameter_control.json"):
        """
        Initialize the ParameterControl instance by loading parameters from a JSON file.

        :param file_path: Path to the JSON file with parameters.
        :type file_path: str
        """
        self.file_path = self._get_absolute_path(file_path)
        file_path = self.file_path.split('\\')
        self.root_path = "\\".join(file_path[:-2])
        self.file_path = "\\".join(file_path[:-2]) + "\\" + file_path[-1:][0]
        self._load_parameters()

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        Convert a relative path to an absolute path based on the project root.

        :param relative_path: The relative path to convert.
        :type relative_path: str
        :return: The absolute path.
        :rtype: str
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(project_root, relative_path)

    def _load_parameters(self) -> None:
        """Load parameters from the JSON file and set them as attributes of the class."""
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Parameter file not found: {self.file_path}")
        
        with open(self.file_path, 'r') as file:
            parameters = json.load(file)
            for key, value in parameters.items():
                if isinstance(value, dict) and 'value' in value and 'encapsulation' in value:
                    encapsulation = value['encapsulation']
                    param_value = value['value']
                    if encapsulation == "private":
                        key = f"__{key}"
                    elif encapsulation == "protected":
                        key = f"_{key}"
                else:
                    param_value = value

                setattr(self, key, param_value)

    def get_path(self, key: str) -> str:
        """
        Get the absolute path of a parameter key that represents a relative path.

        :param key: The key of the parameter representing a path.
        :type key: str
        :return: The absolute path.
        :rtype: str
        """
        if hasattr(self, key):
            relative_path = getattr(self, key)
            relative_path = relative_path.replace('/', '\\')
            return self.root_path + "\\" + relative_path
        else:
            raise AttributeError(f"Parameter '{key}' not found.")

pc = ParameterControl()

class ETL:
    """A class to handle the ETL (Extract, Transform, Load) process."""

    def etl(self, local: bool = True):
        """Run the ETL process."""
        self.extract()
        self.transform()
        return self.load(local=local)

    def extract(self):
        """Extract data from sources."""
        self._configure_env()
        self.download_data()
        self.load_data()

    def transform(self):
        """Transform the data."""
        self.encode()
        self.impute()
        self.process_outliers()
        self.retype()
        self.reformat()

    def load(self, local: bool = True):
        """
        Load the transformed data.

        :param local: If True, load data locally, otherwise save to specified path.
        :type local: bool
        :return: DataFrames of train and test data.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        if local:
            return self.df_train, self.df_test
        else:
            os.makedirs(os.path.expanduser("\\".join(pc.get_path("titanic_frame_train").split('\\')[:-1])), exist_ok=True)

            self.df_train.to_parquet(pc.get_path("titanic_frame_train"), engine='pyarrow')
            self.df_test.to_parquet(pc.get_path("titanic_frame_test"), engine='pyarrow')

    def _configure_env(self):
        """Configure the environment."""
        load_dotenv(dotenv_path=Path(pc.get_path('env_file')))

        if not os.path.isfile("~/.kaggle/kaggle.json"):
            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
            with open(os.path.expanduser("~/.kaggle/kaggle.json"), 'w') as file:
                file.write('{"username":' + os.getenv('KAGGLE_USERNAME') + ',"key":' + os.getenv('KAGGLE_KEY') + '}')

        os.makedirs(os.path.expanduser(pc.get_path("temp_folder")), exist_ok=True)

    def download_data(self):
        """Download the data if not already present."""
        if sum(len(files) for _, _, files in os.walk(pc.get_path("titanic_dataset_raw"))) == 0:
            command = ['kaggle', 'competitions', 'download', '-c', 'titanic', '-p', pc.get_path("temp_folder")]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout.decode())
            if stderr:
                print("Errors:", stderr.decode())

            with zipfile.ZipFile(Path(pc.get_path("titanic_dataset_zip")), 'r') as zip_ref:
                zip_ref.extractall(pc.get_path("titanic_dataset_raw"))

            os.remove(pc.get_path("titanic_dataset_zip"))

    def load_data(self):
        """Load data into DataFrames."""
        self.df_train = pd.read_csv(pc.get_path('titanic_dataset_train'))
        self.df_test = pd.read_csv(pc.get_path('titanic_dataset_test'))

    def encode(self):
        """Encode categorical features."""
        le = LabelEncoder()
        def label_encode(df, column):
            mask = df[column].notnull()
            df.loc[mask, column] = le.fit_transform(df.loc[mask, column])
            df[column] = df[column].astype(float)

        for column in ['Cabin', 'Name', 'Sex', 'Ticket', 'Embarked']:
            label_encode(self.df_train, column)
            label_encode(self.df_test, column)

    def impute(self):
        """Impute missing values using KNN imputation."""
        def KNN_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        self.__dtypes_train = self.df_train.dtypes.to_dict()
        KNN_impute(self.df_train)
        self.__dtypes_test = self.df_test.dtypes.to_dict()
        KNN_impute(self.df_test)

    def process_outliers(self):
        """Process and impute outliers in the data."""
        def impute_outliers(df: pd.DataFrame, columns: Union[List[str], str], lower_threshold: float = 1.5, upper_threshold: float = 1.5) -> pd.DataFrame:
            if isinstance(columns, str):
                columns = [columns]
            
            for column in columns:
                if column in df.columns:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - lower_threshold * IQR
                    upper_bound = Q3 + upper_threshold * IQR
                    
                    median = df[column].median()
                    
                    df[column] = df[column].apply(
                        lambda x: median if x < lower_bound or x > upper_bound else x
                    )
                else:
                    raise ValueError(f"Column {column} not found in the DataFrame.")

        impute_outliers(self.df_train, 'Fare')
        impute_outliers(self.df_test, 'Fare')

    def retype(self):
        """Set correct data types for the DataFrames."""
        def set_dtypes(df, _dtypes):
            for column, _dtype in _dtypes.items():
                df[column] = df[column].astype(_dtype)

        for column in ['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
            self.__dtypes_train[column] = int
            self.__dtypes_test[column] = int

        set_dtypes(self.df_train, self.__dtypes_train)
        set_dtypes(self.df_test, self.__dtypes_test)

    def reformat(self):
        """Reformat the data in the DataFrames."""
        self.df_train['Fare'] = self.df_train['Fare'].round(2)
        self.df_test['Fare'] = self.df_test['Fare'].round(2)

class Analizer:
    """A class to handle data analysis."""

    def analize(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Analyze the train and test data.

        :param train: Training DataFrame.
        :type train: pd.DataFrame
        :param test: Test DataFrame.
        :type test: pd.DataFrame
        :return: Transformed train and test DataFrames.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        self.df_train = train
        self.df_test = test
        return self.__select_features()
        
    def __select_features(self, independent_categorical_columns: List[str] = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], local: bool = True) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
        """
        Select features based on ANOVA significance.

        :param independent_categorical_columns: List of categorical columns to test.
        :type independent_categorical_columns: List[str]
        :param local: If True, return transformed DataFrames, otherwise save to specified path.
        :type local: bool
        :return: Transformed train and test DataFrames if local is True, else None.
        :rtype: Union[Tuple[pd.DataFrame, pd.DataFrame], None]
        """
        def anova_significance(df: pd.DataFrame, columns: List[str], target_column: str, alpha: float = 0.05) -> pd.DataFrame:
            """
            Perform ANOVA test between each categorical column and the target column.

            :param df: The input DataFrame.
            :type df: pd.DataFrame
            :param columns: List of categorical columns to test.
            :type columns: List[str]
            :param target_column: The target column name.
            :type target_column: str
            :param alpha: Significance level (default is 0.05).
            :type alpha: float
            :return: DataFrame with columns: ['Column', 'p-value', 'Significant'].
            :rtype: pd.DataFrame
            """
            results = []

            for column in columns:
                groups = [df[df[column] == level][target_column] for level in df[column].unique()]
                f_stat, p = f_oneway(*groups)
                significant = p < alpha
                results.append({
                    'Column': column,
                    'p-value': p,
                    'Significant': significant
                })

            return pd.DataFrame(results)

        anova_results = anova_significance(self.df_train, independent_categorical_columns, 'Survived')

        if local:
            return self.df_train[[*anova_results.query('Significant')['Column'].to_list(), 'Survived']], self.df_test[[*anova_results.query('Significant')['Column'].to_list()]]
        else:
            self.df_train[[*anova_results.query('Significant')['Column'].to_list(), 'Survived']].to_parquet(pc.get_path("titanic_modeling_train"), engine='pyarrow')
            self.df_test[[*anova_results.query('Significant')['Column'].to_list()]].to_parquet(pc.get_path("titanic_modeling_test"), engine='pyarrow')

def register_model(model, model_name: str, params: Dict[str, Any] = None, metrics: Dict[str, Any] = None, artifacts: Dict[str, str] = None, registry_uri: str = None, tracking_uri: str = None, stage: str = "Staging", verbose: bool = False):
    """
    Register a model in MLflow.

    :param model: The model to register.
    :param model_name: Name of the model.
    :type model_name: str
    :param params: Model parameters.
    :type params: Dict[str, Any]
    :param metrics: Model metrics.
    :type metrics: Dict[str, Any]
    :param artifacts: Additional artifacts.
    :type artifacts: Dict[str, str]
    :param registry_uri: URI for the model registry.
    :type registry_uri: str
    :param tracking_uri: URI for tracking.
    :type tracking_uri: str
    :param stage: Stage to register the model in (default is "Staging").
    :type stage: str
    :param verbose: If True, print additional information.
    :type verbose: bool
    """
    mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")

    if not mlflow.get_experiment_by_name(model_name):
        mlflow.create_experiment(model_name)
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=model_name) as run:
        if params:
            for param_key, param_value in params.items():
                mlflow.log_param(param_key, param_value)
        
        if metrics:
            for metric_key, metric_value in metrics.items():
                mlflow.log_metric(metric_key, metric_value)

        mlflow.sklearn.log_model(model, model_name)

        model_details = mlflow.register_model(f"runs:/{run.info.run_id}/{model_name}", model_name)

        if stage in ["Staging", "Production"]:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_details.version,
                stage=stage
            )

        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path, artifact_name)

        if verbose:
            print("Model saved at:", mlflow.get_artifact_uri(model_name))

def transition_model_to_production(model_name: str, model_version: int, verbose: bool = False):
    """
    Transition a model to the Production stage.

    :param model_name: Name of the model.
    :type model_name: str
    :param model_version: Version of the model.
    :type model_version: int
    :param verbose: If True, print additional information.
    :type verbose: bool
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )
    if verbose:
        print(f"Model {model_name} version {model_version} has been updated to Production.")

@contextmanager
def safe_execution():
    """Context manager for safe execution."""
    try:
        yield
    except Exception as e:
        print("An exception occurred:", e)
        raise e

def easter_egg():
    """Generate a fun fact about the Titanic using GPT-2."""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompts = [
        'How long did it take to build the Titanic?',
        'When did the Titanic sink?',
        'How many people died in the Titanic?',
        'Tell me a fun fact about the Titanic.'
    ]

    input_text = random.choice(prompts)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"GPT2: {generated_text}")

def model_exists(model_name: str) -> bool:
    """
    Check if a model exists in MLflow.

    :param model_name: Name of the model.
    :type model_name: str
    :return: True if the model exists, False otherwise.
    :rtype: bool
    """
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        return len(versions) > 0
    except:
        return False

def print_best_results(grid_search, model_name: str):
    """
    Print the best results from a Grid Search.

    :param grid_search: Grid Search object.
    :param model_name: Name of the model.
    :type model_name: str
    :return: Best estimator from the Grid Search.
    :rtype: Any
    """
    print(f"Best parameters found for {model_name}:", grid_search.best_params_)
    print(f"Best accuracy score for {model_name}:", grid_search.best_score_)
    return grid_search.best_estimator_

class Model(metaclass=Singleton):
    """A class to handle model training, validation, and registration."""

    def __init__(self) -> None:
        self.df_test = None
        self.df_train = None

        self.X_train, self.X_val, self.y_train, self.y_val = None, None, None, None

        self.mlruns_dir = pc.get_path('mlflow_runs')
        self.db_uri = pc.get_path('mlflow_model_db')

        self.models = dict()
        self.searches = dict()

    def load_data(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Load train and test data.

        :param train: Training DataFrame.
        :type train: pd.DataFrame
        :param test: Test DataFrame.
        :type test: pd.DataFrame
        """
        self.df_train = train
        self.df_test = test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.df_train.drop(columns=['Survived']), self.df_train['Survived'], test_size=0.2, random_state=42)

    def train(self, retrain: bool = False):
        """
        Train models.

        :param retrain: If True, retrain the models.
        :type retrain: bool
        """
        if retrain:
            self.logistic_regression(retrain=retrain)
            self.random_forest(retrain=retrain)
            self.svm(retrain=retrain)
            self.xgboost(retrain=retrain)
            self.knn(retrain=retrain)

    def logistic_regression(self, retrain: bool = False):
        """Train Logistic Regression model."""
        if not model_exists('Titanic Model - Logistic Regression') or retrain:
            param_grid = [
                {
                    'logreg__penalty': ['l1', 'l2'],
                    'logreg__C': [0.01, 0.1, 1, 10, 100],
                    'logreg__solver': ['liblinear', 'saga'],
                    'logreg__max_iter': [100, 200, 300, 500]
                },
                {
                    'logreg__penalty': ['elasticnet'],
                    'logreg__C': [0.01, 0.1, 1, 10, 100],
                    'logreg__solver': ['saga'],
                    'logreg__max_iter': [100, 200, 300, 500],
                    'logreg__l1_ratio': [0.5, 0.7, 0.9]
                }
            ]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression())
            ])

            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            grid_search_lr = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', error_score='raise')
            grid_search_lr.fit(self.X_train, self.y_train)

            best_lr = print_best_results(grid_search_lr, "Logistic Regression")
        else:
            print(f"Model 'Titanic Model - Logistic Regression' already exists in MLflow and does not need retraining.")
            client = mlflow.tracking.MlflowClient()
            model_uri = client.get_model_version_download_uri('Titanic Model - Logistic Regression', 1)
            best_lr = mlflow.sklearn.load_model(model_uri)
            print("Model 'Titanic Model - Logistic Regression' loaded from MLflow.")
        
        self.models["Logistic Regression"] = best_lr
        self.searches["Logistic Regression"] = grid_search_lr

    def random_forest(self, retrain: bool = False):
        """Train Random Forest model."""
        if not model_exists('Titanic Model - Random Forest') or retrain:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestClassifier(random_state=42))
            ])

            param_grid = {
                'rf__n_estimators': [100, 200, 300, 500],
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__bootstrap': [True, False]
            }

            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            grid_search_rf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
            grid_search_rf.fit(self.X_train, self.y_train)

            best_rf = print_best_results(grid_search_rf, "Random Forest")
        else:
            print(f"Model 'Titanic Model - Random Forest' already exists in MLflow and does not need retraining.")
            client = mlflow.tracking.MlflowClient()
            model_uri = client.get_model_version_download_uri('Titanic Model - Random Forest', 1)
            best_rf = mlflow.sklearn.load_model(model_uri)
            print("Model 'Titanic Model - Random Forest' loaded from MLflow.")
    
        self.models["Random Forest"] = best_rf
        self.searches["Random Forest"] = grid_search_rf
    
    def svm(self, retrain: bool = False):
        """Train SVM model."""
        if not model_exists('Titanic Model - SVM') or retrain:
            param_grid = [{
                'svc__C': [0.01, 0.1, 1, 10, 100, 200, 500],
                'svc__kernel': ['linear', 'rbf'],
                'svc__gamma': ['scale', 'auto']
                }, {
                'svc__C': [0.01, 0.1, 1, 10, 100, 200, 500],
                'svc__kernel': ['poly'],
                'svc__gamma': ['scale', 'auto'],
                'svc__degree': [3, 4, 5]
                }
            ]

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC())
            ])

            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            grid_search_svm = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
            grid_search_svm.fit(self.X_train, self.y_train)

            best_svm = print_best_results(grid_search_svm, "SVM")
        else:
            print(f"Model 'Titanic Model - SVM' already exists in MLflow and does not need retraining.")
            client = mlflow.tracking.MlflowClient()
            model_uri = client.get_model_version_download_uri('Titanic Model - SVM', 1)
            best_svm = mlflow.sklearn.load_model(model_uri)
            print("Model 'Titanic Model - SVM' loaded from MLflow.")
    
        self.models["SVM"] = best_svm
        self.searches["SVM"] = grid_search_svm
    
    def xgboost(self, retrain: bool = False):
        """Train XGBoost model."""
        if not model_exists('Titanic Model - XGBoost') or retrain:
            param_grid = {
                'xgb__n_estimators': [25, 50, 100, 200, 300, 500],
                'xgb__max_depth': [None, 3, 6, 9],
                'xgb__learning_rate': [0.001, 0.01, 0.1, 0.3],
                'xgb__subsample': [0.8, 1.0],
                'xgb__colsample_bytree': [0.8, 1.0]
            }

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
            ])

            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            grid_search_xgb = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
            grid_search_xgb.fit(self.X_train, self.y_train)

            best_xgb = print_best_results(grid_search_xgb, "XGBoost")
        else:
            print(f"Model 'Titanic Model - XGBoost' already exists in MLflow and does not need retraining.")
            client = mlflow.tracking.MlflowClient()
            model_uri = client.get_model_version_download_uri('Titanic Model - XGBoost', 1)
            best_xgb = mlflow.sklearn.load_model(model_uri)
            print("Model 'Titanic Model - XGBoost' loaded from MLflow.")
    
        self.models["XGBoost"] = best_xgb
        self.searches["XGBoost"] = grid_search_xgb
    
    def knn(self, retrain: bool = False):
        """Train KNN model."""
        if not model_exists('Titanic Model - KNN') or retrain:
            param_grid = {
                'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                'knn__weights': ['uniform', 'distance'],
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ])

            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'f1': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            grid_search_knn = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
            grid_search_knn.fit(self.X_train, self.y_train)

            best_knn = print_best_results(grid_search_knn, "KNN")
        else:
            print(f"Model 'Titanic Model - KNN' already exists in MLflow and does not need retraining.")
            client = mlflow.tracking.MlflowClient()
            model_uri = client.get_model_version_download_uri('Titanic Model - KNN', 1)
            best_knn = mlflow.sklearn.load_model(model_uri)
            print("Model 'Titanic Model - KNN' loaded from MLflow.")
         
        self.models["KNN"] = best_knn
        self.searches["KNN"] = grid_search_knn

    def validate(self):
        """Validate trained models."""
        self.model_scores = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_val)
            self.model_scores[name] = {
                'accuracy': accuracy_score(self.y_val, y_pred),
                'f1': f1_score(self.y_val, y_pred),
                'roc_auc': roc_auc_score(self.y_val, y_pred)
            }

    def registry_models(self, uri: bool = False):
        """Register models in MLflow."""
        for model in list(self.models.keys()):
            metrics = self.model_scores[model]
            
            params = dict({ (k.split('__')[1], v) for k, v in self.searches[model].best_params_.items() })
            best_model = self.models[model]

            register_model(best_model, f"Titanic Model - {model}", params, metrics, registry_uri=self.mlruns_dir, tracking_uri=self.db_uri, stage="Staging", verbose=False)

        if uri:
            print(f'mlflow ui --backend-store-uri sqlite:///{self.db_uri}')

    def publish(self, local: bool = False):
        """Publish the best model to Production stage."""
        best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['accuracy'])
        best_model = self.models[best_model_name]

        transition_model_to_production(f"Titanic Model - {best_model_name}", 1)
        print(f"The best model is: {best_model_name} with the following metrics:")
        print(self.model_scores[best_model_name])

        if local:
            return best_model
