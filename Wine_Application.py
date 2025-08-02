'''
Camille Balo
CS 643
Project 2
Wine Application (can be run in Docker or without Docker)
'''
import sys
import quinn
from pyspark.ml import PipelineModel
from pyspark.sql.session import SparkSession

#command line argument
if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: Wine_Application.py <test_data.csv> <model_directory>")
    sys.exit(1)

filepath = sys.argv[1]
#use second argument if provided, otherwise default to shared NFS directory
modelPath = sys.argv[2] if len(sys.argv) == 3 else "/mnt/shared/model_LogisticRegression"

#create the spark app and run in local mode with all cores
spark = SparkSession.builder.appName("CS643_Project2_App").master("local[*]").getOrCreate()
#set the logging level so that only errors are shown as output on the screen (to clear up the screen for actual app output)
spark.sparkContext.setLogLevel("ERROR")

#load the data from the command line
print(f"\nLoading input data from: {filepath}")
inputData = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load(filepath)
#clean the data
inputData = quinn.with_columns_renamed(lambda s: s.replace('"',''))(inputData)
#rename the quality column to label
inputData = inputData.withColumnRenamed("quality","label")
#display a preview of the data
print("Data loaded and formatted.")
print(inputData.toPandas().head())

#load the saved model from training
print("\nLoading pre-trained model from: ", {modelPath})
model = PipelineModel.load(modelPath)

#run the predictions using the loaded model
print("\nRunning wine predictions...")
predictions = model.transform(inputData)
#display the results
print("\nResults:")
predictions.select("label", "prediction").show(truncate=False, n=predictions.count())




