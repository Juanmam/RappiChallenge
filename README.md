# RappiChallenge

## Notebooks
En la seccion de notebooks encontraran 3 notebooks para ejecutar, similar a lo que hace la consola pero de hecho programe todo primero en el notebook ya que permite un prototipado mas agil y despues lo pase al package, que lo encontraran bajo la carpeta src.

- ETL: Un notebook disenado para configurar el ambiente de Kaggle, descargar la datos a bronze, transformar los datos, y cargar los datos transformados a silver.

- Analisis: Incluye dos reportes usando la libreria Dataprep.ai, la cual ayuda a entender el perfil de los datos. No solo eso, tambien incluye una seccion de exploracion de significancia, en donde usamos diferentes metodos estadisticos como el ANOVA, ANCOVA y el Chi-Squared para ayudarnos en la seleccion de caracteristicas.

- Model: Este notebook incluye el entrenamiento de 5 modelos estadisticos para la tarea de clasificacion, los cuales son entrenados por medio de un gridsearch para aproximar los mejores parametros de cada modelo. Despues, se realiza una verificacion cruzada con un subconjuto de los datos. Por ultimo se realiza el registro de los modelos y el paso a produccion de el mejor de estos.

### Metricas

Las metricas que decidi utilizar en general fueron:

- Accuracy: Es una metrica muy generica, nos da una idea del numero de predicciones correctas en relacion con el total del conjunto de datos. Nos deja por fuera muchos factores como, que tan bien manejamos falsos positivos o desbalances.

- F1 Score: Es una metrica muy buena por que relaciona harmonicamente la precision y el recall, lo cual nos ayuda a tener ambos en cuenta y a amortiguar el impacto de outliers dentro del set.

- ROC_AUC: Es muy util para entender el comportamiento del modelo a pesar del desbalance de datos. En este caso, tenemos un ligero desbalance de clases de 60:40.

Nota: En cuanto al desbalance de datos, no considero que amerite tencicas de balanceo como SMOTE o SMOTENC dado a que la diferencia de clases es solo de 10%, asi que F1 Score y Accuracy deberian hacer un buen trabajo.

De estos metodos, seleccione Accuracy como el principal, a pesar de evaluar el resto de igual manera, principalmente por que no considero que para el tamano del dataset se amerite algo diferente. En otro escenario, F1 Score honestamente hubiera sido una mejor seleccion, a pesar que en los resultados que obtuvimos fue la metrica mas baja.

### Resultados

El mejor modelo en funcion de la data de validacion fue el KNN, con las siguientes metricas. 

- Accuracy: 0.8156424581005587
- F1 Score: 0.7659574468085106
- ROC_AUC:  0.802960102960103

De esta data podemos concluir que en terminos generales, el modelo no es ni el mejor, ni el peor, acertamos en un 81.5% de los registros de validacion, como pueden observar la diferencia entre ROC_AUC y accuracy no es significativa por lo que podemos concluir que el impacto del desbalance de clases no era tan relevante, y con el F1 Score podemos entender el comportamiento del model en funcion de los Falsos Positivos y de los Falsos Negativos, siendo un punto medio entre entender los Falsos Positivos con la precision y entender los Falsos Negativos con el Recall.

Lo considero bueno? Es debatible bueno en que contexto, en el de la prueba tecnica, probablemente sea bueno, en el de un modelo PoC, probablemente tambien sea bueno para algunos sectores de negocio, pero para un producto en desarrollo con varias versiones? Definitivamente no.

### Importancia de Caracteristicas

Este va a ser un tema dificil de hablar si quiero pasar esta prueba tecnica, pero debo explicar los resultados observados del modelo y de las correlaciones encontradas en la data. A continuacion mostrare el scoring de importancia de cada caracteristica:

- Sex	    0.165549
- Pclass	0.036313
- Parch	    0.027374
- Embarked	0.025512
- SibSp	    0.024022
- Age	    0.017877
- Cabin	    0.010615
- Ticket	0.009870

A continuacion haremos un analisis del top 5 de las caracteristicas mas importantes en orden:

- Sex: En caso de emergencia de naufragio, existen dos politicas muy conocidas, el "Salvese quien pueda" y el "Mujeres y niños primero". La primera politica se usaba mas en la antiguedad, ya a medida que el transito transatlantico se normalizo y con el naufragio del RMS Birkenhead en 1852, se empezo a normalizar el uso de "Mujeres y niños primero".
- Pclass: La mayor parte de primera clase estaba en la seccion central del barco, area en donde estaban los botes salvavidas, mientras que el impacto y la ruptura del casco, afecto principalmente a una poblacion considerable de segunda y tercera clase. No solo si eras de una clase peor, te tocaba andar mas para llegar a los salvavidas, pero tambien fueron los primeros afectados de la colision, en especial tercera clase.
- Parch: El hecho de que las personas tuvieran que ir en busqueda de hijos o padres que por su edad no tenian la misma capacidad, llevaba a mucha gente a perder tiempo preciado para llegar a los salvavidas.
- Embarked: Teniamos tres lables diferentes, cada una indicando un puerto diferente en un pais diferente. Estamos hablando de un puerto Ingles, uno Irlandes y otro Frances. Al hacer un analisis breve, consultando el porcentaje de personas en una clase en funcion de su lugar de origen, podemos encontrar cosas muy interesantes. Los Franceses eran el 69% de primera clase a pesar de ser solo 168 pasajeros vs 700+ que habia en las otras dos categoricas. Los Irlandeses se llevaron la corona en tercera clase, con un 50%, mientras que los ingleses se llevaron la mayoria de segunda clase con 64%. La ubicacion donde embarcaron no les dio desventajas geneticas ni de fenotipo, pero les dio desventajas de clases, y como lo mencione anteriormente, la clase era fundamental por la ubicacion de los camarotes y el contexto del incidente.
- SibSp: Similar a lo que pasaba con los padres e hijos, pero con los hermanos y hermanas era mas por mantener el nucleo familiar unido, al fin de cuentas, no importa cuanto te fastidie tu hermana mayor, dejarla morir en las aguas del atlantico norte no es muy recomendable.

### Implementacion

En caso de que queramos poner este modelo en produccion, la forma mas facil, asi como esta, seria empaquetando todo en un contenedor de docker, y utilizar algun servicio como AKS o ACR para hostear el contenedor. Ya si se desea invertir mas tiempo en esto, se puede pasar esta implementacion a un ambiente de Databricks, que no solo se integra muy bien con MLFlow, pero tambien tiene mecanismos de gobierno de datos y hosting de modelos, y dado a que no es un modelo critico como tal, el tiempo de respuesta es un factor secundario y por lo contrario puede ayudar a reducir costos. En caso de que se vaya por un AKS o ACR, se puede usar la API de AML para registrar los modelos alli, y hasta se podria usar un blob storage en frio con el fin de almacenar versiones de la data, reduciendo costos de versionamiento en un Data Lake. 

Se puede usar DevOps para el despliegue continuo, Terraform o Bicep para definir el IaC. Esto incluye el manejo de artefactos de codigo por medio del git integrado, el uso de pipelines para los diferentes procesos, incluyendo re-entrenamiento y creacion del ambiente y tambien se puede aprovechar el uso de boards para el equipo de gestion de proyectos. Para este punto, se recomienda usar un Key Vault para almacenar llaves y valores secretos. 

Se puede implementar un sistema de control de cargas independiente al sistema de control de parametros establecido en una base de datos SQL y se puede robusteser mucho mas la implementacion de control de parametros que hice, y usar una base de datos tipo Cosmos DB o de Redis para almacenamiento de estos parametros. La parte de carga se puede remplazar por un Fabric o por un DataFactory para la ingesta y transformacion de datos. 

Se recomienda tambien usar algun servicio tipo API Managment en combinacion con una subnet y un Application Gateway para gobernar los entrypoints, dar una capa de balanceo y proteger las unidades de computo de posibles ataques.

### Practicas
- Estandarizacion del manejo de excepciones. Se puede usar esta tecnica modular para facilmente logear errores por medio del context manager disenado.
- Uso de arquitectura medallon, para categorizar la data en funcion de su madurez.
- Uso de un control de parametros simple, en parte es un pseudo control de cargas, pero esta disenado actualmente para almacenar tambien parametros en general, no solo rutas.
- Documentacion, usando sphynx, que puede usarse para generar documentacion automaticamente.

## Configurations
Antes de ejecutar el codigo, se deben llenar los datos necesarios en el archivo .env y el archivo parameter_control.json. Estos archivos los pueden encontrar en la carpeta raiz y funcionan como archivos de configuracion que se cargan durante la ejecucion del codigo. El archivo .env, se utiliza para cargar los datos de Kaggle como variables de ambiente que pueden ser usadas en la ejecucion, mientras que el parameter_control.json, permite configurar comportamientos del programa, como donde se guardan los artefactos y demas variables que pueden ser utiles para la ejecucion.

.env
~~~
KAGGLE_USERNAME="mi_usuario"
KAGGLE_KEY="mi_llave"
~~~

Nota: Para mas informacion en como encontrar esta informacion de autenticacion, pueden referirse a la [documentacion](https://www.kaggle.com/docs/api).

parameter_control.json
~~~
{
    "env_file": ".env",
    "temp_folder": ".temp",
    "titanic_dataset_zip": ".temp/titanic.zip",
    "titanic_dataset_raw": "data/bronze/proyecto_titanic",
    "titanic_dataset_train": "data/bronze/proyecto_titanic/train.csv",
    "titanic_dataset_test": "data/bronze/proyecto_titanic/test.csv",
    "titanic_frame_train": "data/silver/proyecto_titanic/train.parquet",
    "titanic_frame_test": "data/silver/proyecto_titanic/test.parquet",
    "titanic_modeling_train": "data/silver/proyecto_titanic/train_model_data.parquet",
    "titanic_modeling_test": "data/silver/proyecto_titanic/test_model_data.parquet",
    "mlflow_runs": ".mlruns",
    "mlflow_model_db": ".mlflow.db"
}
~~~

Estos son los parametros requeridos por el programa, sus valores pueden ser modificados en funcion de la raiz del proyecto.

## Usage
En la raiz del proyecto encontraran un archivo llamado main.py, el archivo principal del proyecto. Hay dos formas de utilizarse:

```
python main.py [command] [args]* [--[param_name] [value]]*
```

Tambien pueden correr la terminar interactiva por medio de

```
python main.py
Entering interactive mode. Type 'exit' to quit, 'help' for list of commands.
> [command] [args]* [--[param_name] [value]]*
```

Se puede usar el output de las funciones ya ejecutadas como en el siguiente ejemplo:

```
python main.py
Entering interactive mode. Type 'exit' to quit, 'help' for list of commands.
> add 1 1
2
> square add.output
4
```

El proceso entero se puede correr usando los siguientes comandos:

```
python main.py
Entering interactive mode. Type 'exit' to quit, 'help' for list of commands.
> etl
> analize etl.output
> model_data analize.output
> train --retrain True
> validate
> registry_models
> publish
```

### Comandos

Esta es una lista de los comandos disponibles para ejecutar:

- full_process: Ejecuta todas las siguientes funciones en orden, publicando al final el mejor modelo.
- etl: Corre el proceso de ETL, descargando la data de kaggle y guardandola al local.
- analize: Ejecuta la seleccion de caracteristicas por medio de un analizis de significancia estadistica usando como metodo ANOVA.
- model_data: Carga la data para el consumo del modelo.
- train: Entrena los modelos de: Logistic Regression, Random Forest, SVM, XGBoost y KNN. Si ya se cargaron anteriormente a MLFlow usando registry_models, se puede usar el parametro --retrain True para re-entrenarse.
- validate: Selecciona el mejor modelo al realizar una validacion de los modelos generados, por medio de un subset de navegacion.
- registry_models: Carga los modelos, metricas y parametros a MLFlow.
- publish: Publica el mejor modelo como un modelo de produccion, dejando el resto en staging.

Si necesitan acceder a la UI de MLFlow, lo pueden hacer por medio de

```
mlflow ui --backend-store-uri sqlite:///[db_uri]
```

donde db_uri es la ubicacion relativa que se definio para el archivo. Se recomienda usar un archivo sqlite para hacer el ejercicio mas practico/rapido.


## Testing

El archivo test.py contiene las pruebas para las clases de ETL, Analizer y Model. El reporte de coverage indica que las pruebas apenas cubren un 40% del codigo, pero teniendo en cuenta que este 40% cubre las funcionalidades basicas del probrama que se ejecutas por comandos, no considero que sea tan critico alcanzar para este caso un numero mayor. 

El porcentaje de coverage que veran en el reporte para cada archivo, indica que tan cubierto de errores, o al menos, que tan probado esta la totalidad del codigo en funcion de las pruebas disenadas. Por ejemplo, para el caso de misc, tenemos una covertura del 31%, pero dentro de este 31% estan los metodos que se utilizan para entrenamiento, que redundamente llaman otros metodos que no estan en coverage ya que sirven como wrappers.

Para generar el archivo, solo deben ejecutar los siguientes comandos en consola

```
coverage run -m unittest discover
```

Despues de correr las pruebas, generamos el reporte

```
coverage report -m
```

Lo podemos guardar en un archivo HTML para que sea mas facil de consumir visualmente.

```
coverage html
```


Tambien podemos abrir el archivo HTML que generamos al correr lo siguiente:

```
start htmlcov/index.html
```