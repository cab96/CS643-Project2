'''
Camille Balo
CS 643
Project 2
Model Training + Saving Model

Models Trained:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Naive Bayes
'''
import quinn
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

#create spark sessin and application name
spark = SparkSession.builder.appName("CS643_Project2_Training").getOrCreate()
#set the spark logging level so that only ERRORs are shown--this just helps to clean up the output on the screen
spark.sparkContext.setLogLevel("ERROR")

#pull the training and validation csv files
#this path and files must also exist on the workers so they can pull the data as well
trainingData = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ec2-user/Project2/TrainingDataset.csv')
validationData = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ec2-user/Project2/ValidationDataset.csv')

print("\nSuccessfully loaded data.")
print("Formatting data.")
#remove the double quotes from the column names
#clean / format the training data
trainingData = quinn.with_columns_renamed(lambda s: s.replace('"',''))(trainingData)
#rename the quality column to label
trainingData = trainingData.withColumnRenamed('quality', 'label')
#clean /format the validation data
validationData = quinn.with_columns_renamed(lambda s: s.replace('"',''))(validationData)
#rename the quality column to label
validationData = validationData.withColumnRenamed('quality', 'label')
#show a preview of the formatted data
print("Data has been formatted.")
print(trainingData.toPandas().head())

#create a vector for the columns that are the input
columnVector = VectorAssembler(inputCols=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"], outputCol="inputFeatures")
#normalixe columnVector
normal = Normalizer(inputCol="inputFeatures", outputCol="features")

#these are the models I will be testing
lr = LogisticRegression()
LRpipe = Pipeline(stages=[columnVector, normal, lr])

dt = DecisionTreeClassifier()
DTpipe = Pipeline(stages=[columnVector, normal, dt])

rf = RandomForestClassifier()
RFpipe = Pipeline(stages=[columnVector, normal, rf])

nb = NaiveBayes()
NBpipe = Pipeline(stages=[columnVector, normal, nb])

#create parameter grid for cross validator
crossvalGrid = ParamGridBuilder().build()
#F1 will be used as the evalution method
f1Eval = MulticlassClassificationEvaluator(metricName="f1")

#logistic regression
crossValid = CrossValidator(estimator=LRpipe, estimatorParamMaps=crossvalGrid, evaluator=f1Eval, numFolds=3)
print("\nBegining training with LogisticRegression Model.")
LRmodel = crossValid.fit(trainingData) 
print("F1 Score for LogisticRegression Model: ", f1Eval.evaluate(LRmodel.transform(validationData)))
print("\nSample predictions from LogisticRegression Model:")
print(LRmodel.transform(validationData).select("label", "prediction").show(10))

#decision tree classifier
crossValid = CrossValidator(estimator=DTpipe, estimatorParamMaps=crossvalGrid, evaluator=f1Eval, numFolds=3)
print("\nBeginning training with DecisionTreeClassifier Model.")
DTmodel = crossValid.fit(trainingData)
print("F1 Score for DecisionTreeClassifier Model: ", f1Eval.evaluate(DTmodel.transform(validationData)))
print("\nSample predictions from DecisionTreeClassifier Model:")
print(DTmodel.transform(validationData).select("label", "prediction").show(10))

#random forest classifier
crossValid = CrossValidator(estimator=RFpipe, estimatorParamMaps=crossvalGrid, evaluator=f1Eval, numFolds=3)
print("\nBeginning Training with RandomForestClassifier Model.")
RFmodel = crossValid.fit(trainingData) 
print("F1 Score for RandomForestClassifier Model: ", f1Eval.evaluate(RFmodel.transform(validationData)))
print("\nSample predictions from RandomForestClassifier Model:")
print(RFmodel.transform(validationData).select("label", "prediction").show(10))

#naive bayes
crossValid = CrossValidator(estimator=NBpipe, estimatorParamMaps=crossvalGrid, evaluator=f1Eval, numFolds=3)
print("\nBeginning training with NaiveBayes Model.")
NBmodel = crossValid.fit(trainingData)
print("F1 Score for NaiveBayes Model: ", f1Eval.evaluate(NBmodel.transform(validationData)))
print("\nSample predictions from NaiveBayes Model:")
print(NBmodel.transform(validationData).select("label", "prediction").show(10))

#store the scores in a dictionary to determine which is the highest
scores = {
    "LogisticRegression": f1Eval.evaluate(LRmodel.transform(validationData)),
    "DecisionTreeClassifier": f1Eval.evaluate(DTmodel.transform(validationData)),
    "RandomForestClassifier": f1Eval.evaluate(RFmodel.transform(validationData)),
    "NaiveBayes": f1Eval.evaluate(NBmodel.transform(validationData)),
}

#print scores
print("\nFinal F1 Scores:")
for modelName, score in scores.items():
    print(f"{modelName}: {score:.4f}")

#determine the best model by which F1 score value is the highest
bestModel = max(scores, key=scores.get)
print(f"\nBest Model (highest F1 score): {bestModel}")

#save the model to load later in the application
savePath = f"/mnt/shared/model_{bestModel}"
#evaluate which is the best model and save it
if bestModel == "LogisticRegression":
    LRmodel.bestModel.write().overwrite().save(savePath)
elif bestModel == "RandomForestClassifier":
    RFmodel.bestModel.write().overwrite().save(savePath)
elif bestModel == "DecisionTreeClassifier":
    DTmodel.bestModel.write().overwrite().save(savePath)
elif bestModel == "NaiveBayes":
    NBmodel.bestModel.write().overwrite().save(savePath)
print(f"{bestModel} saved to: {savePath}")