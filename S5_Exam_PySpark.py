# Import de SparkSession et de SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext

# Création d'un SparkContext
sc = SparkContext.getOrCreate()

# Création d'une session Spark
spark = SparkSession \
    .builder \
    .appName("Exam PySpark SCH") \
    .getOrCreate()
        
spark

# Importer le fichier creditcard.scv
df_raw = spark.read.csv('data/creditcard.csv', header=True)

# Afficher un extrait du DataFrame df_raw
df_raw.sample(False, .001, seed = 222).toPandas()


##########################
distinct_classes = df_raw.select('Class').distinct()

# Convertir le résultat en DataFrame Pandas pour l'affichage
distinct_classes_df = distinct_classes.toPandas()
print(distinct_classes_df)

##########################

# importer col
from pyspark.sql.functions import col
    
#Créer un DataFrame df à partir de df_raw en changeant les colonnes des variables explicatives en double
#et la variable cible, Class, en int.
exprs = [col(c).cast("double") for c in df_raw.columns[1:30]]

df = df_raw.select(df_raw.Class.cast('int'),
                           *exprs)

# Afficher le schema
print("Schema du DataFrame df :")
df.printSchema()


# supprimer les lignes manquantes: 
df = df.dropna()

# Importer la focntion DenseVector
from pyspark.ml.linalg import DenseVector
# Conversion de la base de données au format svmlib
#Créer un rdd rdd_ml séparant la variable à expliquer des features (à mettre sous forme DenseVector)
rdd_ml = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

#Créer un DataFrame df_ml contenant notre base de données sous deux variables : 'labels' et 'features'
df_ml = spark.createDataFrame(rdd_ml, ['label', 'features'])

df_ml.show()

# Créer deux DataFrames appelés train et test contenant chacun respectivement 80% et 20% des données
train, test = df_ml.randomSplit([0.8, 0.2], seed=222)

# Import de la fonction RandomForestClassifier 
from pyspark.ml.classification import RandomForestClassifier
# Créer un classificateur Forest cf
clf = RandomForestClassifier(featuresCol='features', labelCol='label', predictionCol='prediction',seed = 222)

# Entraîner le modèle sur l'ensemble d'entraînement
model = clf.fit(train)

# Faire des prédictions sur l'ensemble de test
predictions = model.transform(test)

# Affichage d'un extrait des prédictions 
predictions.sample(False, 0.001 , seed = 222).toPandas()

# Import d'un évaluateur MulticlassClassificationEvaluator du package pyspark.ml.evaluation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Création d'un évaluateur 
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Calcul et affichage de la précision du modèle 
accuracy = evaluator.evaluate(predictions)
print(accuracy)

spark.stop()