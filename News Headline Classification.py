#This script aims to classify news articles, based on the effect they have on a company's share prices.
#Articles are classifed on a scale from 2 to -2, where 2 is a strongly positive effect, and -2 is a strongly negative effect.
#The script usesPySpark machine learning.

import findspark

findspark.init('/opt/spark-2.1.2-bin-hadoop2.7')

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Testing") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, NGram
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


def news_classifier():

    data = spark.read.option("mode", "DROPMALFORMED").load("/news_data.csv", format="csv", header="true", inferSchema='true')

    data.first()
    data.printSchema()

    #There is a field in the data called constituent_id, which is basically the company which the news headline is about. We want to drop that column from our data.
    drop_list = ['constituent_id']

    data = data.select([column for column in data.columns if column not in drop_list])

    data.show(5)

    data.printSchema()

    # regular expression tokenizer
    regexTokenizer = RegexTokenizer(inputCol="news_title", outputCol="words", pattern="\\W")

    # remove stop words
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")

    #compute bigrams
    ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")

    # Add HashingTF and IDF to transformation
    hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

    #string indexer
    label_stringIdx = StringIndexer(inputCol = "weekly_returns", outputCol = "label")

    #create processing pipeline
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, ngram, hashingTF, idf, label_stringIdx])

    # Fit the pipeline to training data.
    pipelineFit = pipeline.fit(data)
    dataset = pipelineFit.transform(data)

    dataset.show(5)

    # Randomly split data into training and test sets. set seed for reproducibility
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
    print("Training Dataset Count: " + str(trainingData.count()))
    print("Test Dataset Count: " + str(testData.count()))

    # Build a Logistic Regression model
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0, family="multinomial")

    # Train model with Training Data
    lrModel = lr.fit(trainingData)

    predictions = lrModel.transform(testData)

    predictions.filter(predictions['prediction'] == 0) \
        .select("news_title","weekly_returns","probability","label","prediction") \
        .orderBy("probability", ascending=False) \
        .show(n = 10, truncate = 30)

    #multiclass evaluator
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    print(evaluator.evaluate(predictions))

    #save predictions to csv

    predictions = predictions.select("news_title", "weekly_returns", "prediction")
    predictions.write.format("csv").save("/Desktop/predictions-spark.csv")

    #save machine learning model
    model_path = "/Desktop/Spark_Model"
    lrModel.save(model_path)

    #load model again, to make sure it works
    ml_model = lrModel.load(model_path)
    predictions2 = ml_model.transform(testData)

    #make predictions with loaded model
    predictions2.filter(predictions2['prediction'] == 0) \
        .select("news_title","weekly_returns","probability","label","prediction") \
        .orderBy("probability", ascending=False) \
        .show(n = 10, truncate = 30)

    #end spark session
    spark.stop()

if __name__ == '__main__':
    news_classifier()



