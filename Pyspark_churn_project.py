# region Importing
#Initialize Spark
import findspark
findspark.init()
from pyspark import SparkContext
sc = SparkContext()
from pyspark.sql import SQLContext
sql = SQLContext(sc)
#Import sfrom ql
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.sql.functions import coalesce
#Import from ml
from pyspark.ml.feature import  StringIndexer, VectorAssembler, OneHotEncoder,StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.tuning import ParamGridBuilder
#import from mllib
from pyspark.mllib.evaluation import MulticlassMetrics
#import stratified CV
from spark_stratifier import StratifiedCrossValidator
#import from python libraries
from sklearn.model_selection import train_test_split
import numpy as np
# endregion

# region Data loading and cleaning
df = sql.read.csv('./DATA/Retention Data/Telco-Customer-Churn.csv',inferSchema=True,header=True)
df=df.withColumn("TotalCharges", df["TotalCharges"].cast(DoubleType()))
#New customers (tenure=0) have missing in total charges
#df.where(col("TotalCharges").isNull()).show()
#fill missing values by copying montly charghes into total charges
df=df.withColumn("TotalCharges",coalesce(df.TotalCharges,df.MonthlyCharges))

#df.filter(col("tenure")=="0").show()
# endregion

# region Data splitting
#Stratified splitting through pandas

df_pd=df.toPandas()

df_pd.Churn[df_pd.Churn=="Yes"]=1
df_pd.Churn[df_pd.Churn=="No"]=0
df_pd.Churn=df_pd.Churn.astype(bool)
training, test = train_test_split(df_pd, test_size=0.3, stratify=df_pd.Churn,random_state=123)

training = sql.createDataFrame(training,schema=df.schema)
test= sql.createDataFrame(test,schema=df.schema)
# endregion

# region Pre processing
stages = []
numeric_cols = ["tenure","MonthlyCharges","TotalCharges"]

for n in numeric_cols:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[n],outputCol=n+"_vect")
    scaler = StandardScaler(inputCol=n+"_vect", outputCol=n+"_scaled",withMean=True, withStd=True)
    stages += [assembler, scaler]

categorical_cols= ['gender','SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', ]

for categoricalCol in categorical_cols:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

labelIndexer = StringIndexer(inputCol = 'Churn', outputCol = 'label')
stages += [labelIndexer]

assemblerInputs = [c + "classVec" for c in categorical_cols]+ [n +"_scaled" for n in numeric_cols]
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

#Pipeline
pipeline = Pipeline(stages = stages)
training.cache()
train_preprocess = pipeline.fit(training)
training.unpersist()
training = train_preprocess.transform(training)

test.cache()
test_preprocess = pipeline.fit(test)
test.unpersist()
test = train_preprocess.transform(test)
# endregion

#Save pipeline
#pipelinePath = './RF_Pipeline'
#train_preprocess.write().overwrite().save(pipelinePath)
#

#Load pipline
from pyspark.ml import PipelineModel
#loadedPipelineModel = PipelineModel.load(pipelinePath)

# region Crossvaldiation settings
rf=RandomForestClassifier(labelCol="label",featuresCol="features",numTrees=200,seed=123)
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth , [4,5,6,7]).addGrid(rf.featureSubsetStrategy , [str(x) for x in np.linspace(start = 0.14, stop =0.23 ,
                                                                                                                               num=4)]).build()

#User defined metric
#F2score

class f2_eval(Evaluator):

    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):

        df_rdd=dataset.select("label","prediction").rdd
        predictionAndLabel=df_rdd.map(lambda d: (d.prediction,d.label))
        metric= MulticlassMetrics(predictionAndLabel)
        return metric.fMeasure(label=1.0,beta=2.0)

    def isLargerBetter(self):
        return True

# endregion


# region Crossvalidation
#Stratified cross-validation through external library
cv = StratifiedCrossValidator(estimator=rf,estimatorParamMaps=paramGrid,evaluator=f2_eval(), numFolds=5, seed=123,parallelism=8)
training.cache()
cvModel = cv.fit(training)
training.unpersist()
best_model = cvModel.bestModel
# endregion


# region Save
cvModel.bestModel.write().overwrite().save("./Pyspark_CV_model")
# endregion

# region load
from pyspark.ml.classification import RandomForestClassificationModel
best_model = RandomForestClassificationModel.load("./Pyspark_CV_model")
# endregion


#Print parameters
{param[0].name: param[1] for param in best_model.extractParamMap().items()}

# region Optimal threshold
#Should have used crossvalidation but it takes some time to implement it on pyspark

proba_and_label_train = best_model.transform(training).select("probability","label").rdd
thresholds=np.round(np.arange(0.10,0.90,0.025),3)

predi_thresh=[]

for i in range(len(thresholds)):
    predi_thresh=np.append(predi_thresh, proba_and_label_train.map(lambda x: (float(x[0][1]>= thresholds[i]),x[1])))

f_scores=[]
for i in range(len(thresholds)):
    predi_thresh[i].cache()
    multi_metrics = MulticlassMetrics(predi_thresh[i])
    f_scores= np.append(f_scores,multi_metrics.fMeasure(label=1.0, beta=2.0))
    predi_thresh[i].unpersist()

opti_thresh=thresholds[np.argmax(f_scores)]

# endregion

# region test prediction and score
proba_and_label_test = best_model.transform(test).select("probability","label").rdd
predi_and_label_test=proba_and_label_test.map(lambda x: (float(x[0][1]>= opti_thresh),x[1]))
multi_metrics = MulticlassMetrics(predi_and_label_test)
#multi_metrics.fMeasure(label=1.0,beta=2.0)
#score=0.746
#multi_metrics.confusionMatrix().toArray()
# endregion