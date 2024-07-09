This repository is note from watching video [Google Cloud Machine Learning Engineer Certification Prep](https://www.udemy.com/course/google-cloud-machine-learning-engineer-certification-prep/).

Building models in google cloud:
+ Cloud AutoML
+ AI platform training
+ Kubeflow
+ Dataproc and Spar kML
+ BigQuery ML

To deploying ML model, we need:
+ ML services
+ Containers
+ Orchestration

Step to production:
+ Wrap our model in a service > endpoint as output
+ Containerization
+ Deploy and orchestration > google cloud provides Cloud Run and Kubernetes Engine for running containers. Cloud Run good for small number of containers.

Comprehensive ML platforms: Kubeflow and VertexAI.

**Service in VertexAI**:
+ Datasets
+ Feature store
+ Workbences > manage notebook and user manage notebook. In user manage notebook, we can setting the environment.
+ Training > we can use AutoML or custom training. Custom training usually use for models at scale.
+ Cloud storage > really well-suited for unstructured data (image, video, text). We can also used in applications, for example in microservices architecture for temporary storage between services. 
+ BigQuery: serverless data warehouse, petabyte scale, uses SQL but is not a relational database, analytical database.
    + Datasets ㅡ collection of tables and views.
    + Tables ㅡ supports scalar and nested structures, stored in columnar format, and partitioning.
    + Views ㅡ projection of one or more tables, tables can be joined, and views can be materialized.  

    **Partitioned tables**: table divided into segments (partitions). It will improves query performance, and lowers cost.

    **Cluestered tables**: is another optimization we can use with BigQuery. It is data sorted based on values in one or more columns. It can improve performance of aggregate of aggregate queries. It also can reduce scanning when cluster columns used in WHERE clause.

+ Dataflow and Dataproc: is designed for batch processing of large datasets. 
    + **Dataflow** > horizontally scalable, managed service, and supports windowing operations. Windowing operations especially important when we working with streaming data and time series data. 
    + **Dataproc** > managed Spark and Hadoop service. Dataproc often used for ETL and ELT. When we use Dataproc, we often create ephemeral clusters. Epheremal clusters means we start cluster when we need we run a job and then we shutdown the cluster.

**Virtual machines**:
+ Compute engine > we can configure virtual machine.
+ Managed instance groups allow us to work with multiple identical virtual machines.
+ Containers > Cloud Run and Kubernetes Engine.
    + **Cloud Run**: 
        + We can run containers in two modes, service and batch. Service mode for a machine learning algorithn that provides predictions. A containers always available, waiting us to call the API. Meanwhile, batch mode useful for thins like ETL.
        + Use for mircoservices and endpoints.
        + Use when minimizing administration is a priority.
    + **Kubernetes Engine**:
        + Large scale container deployments.
        + We have control over cluster.        

GPUs and TPUs: GPUs is for higher precision than TPUs.

**Training machine learning**: 
+ Hyperparameter tuning: please remember that hyperparameters are not learned! "Tune" means finding which combinations are the best. Methods that we can use include Grid Search, Random Search, and Bayesian Search. Bayesian Search is sequential model-based optimization. It use previous iteration to improve current iteration.
+ Unit tests: tests that run automatically in the CI/CD pipeline to prevent deploying an broken model. For example, feature engineering functionality, encoding inputs, custom modules, and output types.
+ **Distributed training**: training a model across multiple nodes in a cluster.
    + Available in VertexAI
    + Need to use a framework that supports distributed training, like TensorFlow.
    + Role of nodes in distributed training:
        + Primary replica > to manage other nodes.
        + Workers > to do a portion of the training work.
        + Parameter services > to store model parameters and coodinate shared model state across workers.
        + Evaluators > to evaluate models.

    Reduce training time with reduction server allow us to communicating gradients (when training model) between nodes. Reduction server requires use of GPUs. 

**Serving options**: we can use pre-built containers like TensorFlow, TensorFlow Optimized Runtime, Scikit-learn, and XGBoost. We can optionally configure those machines including use GPUs, number of Virtual CPUs per node, and memory per node. We also can use custom containers. The requirements for a custom container with VertexAI:
+ Provide a docker container image running HTTP server.
+ Responds to health checks, liveness checks, and predicition requests.
+ Each request and response must be 1.5MB or smaller.
+ Vertex Service AI Agent is Google managed service account, has sufficient permissions to work with custom containers.

Other example of pre-build containers:
+ NVIDIA Triton is an open source inference serving platform optimized for CPUs and GPUs. VertexAI Prediction runs in custom container published by NVIDIA. These supports TensorFlow, PyTorch, TensorRT, Scikit-learn, and XGBoost. 
+ Optimized TensorFlow Runtime allow us to run TensorFlow models at lower cost and latecy than open source pre-built TensorFlow containers.

**Prediction services**: VertexAI allocates nodes for online and batch predictions. Online prediction (synch) have endpoint, and batch prediction (asynch) run as jobs.

**Monitoring**: VertexAI model monitors predicition input data fro skew and drift. **Skew** ㅡ feature data distribution in production deviates from training. **Drift** ㅡ feature data distribution in production changes significantly over time. Scope of monitoring: supports skew and drift detection for categorical and numerical features. Skew based on training data, meanwhile drfit based on recent past production data. When distance score between distributions exceed specified thresold identify as skew or drift. 

**Optimizing pipeline**: 
+ Data processing:
    + If our data already in BigQuery and we use TensorFlow, we can use the BigQueryClient to access the data.
    + Keep in mind with BigQuery, our costs are in part based on how much data you scan. So use partition tables so we can minimize the amount of data scanned. 
    + If we have some tolerance on time, and we don't need like super low latency for processing the data, we can use Dataflow FlexRS (flexible resource) scheduling.
    + Dataflow Shuffle is an operation where we have to move data between nodes or servers.
    + If we use Tensorflow, and we pre-processing data, we can save the pre-processing result data as TFRecord.
+ Model training:
    + Distributed training is really useful when we build really large model, but before we depend on distributed training, consider scalling up the size of our machine. 
    + If we use TensorFlow, be sure to use MultiWorkMirroredStrategy in TF distributed training that optimizes data sharing among the different nodes.
    + Use tf.data API to maximize efficiency of data pipelines using GPUs and TPUs. 
    + Stream data from cloud storage for Scikit-learn models instead if copying data to servers.
    + Use early stopping.
    + Use automatic hyperparameter tuning.
+ Model serving:
    + Use reduced-precision floating types, like use 16-bit instead of 32-bit. Use mixed-precision during training for numeric stability.
    + Use post-training quantization, like reduces model size.
    + Use TensorFlow Model Optimization Toolkit, if we use TensorFlow.
    + If we use NVIDIA GPUs, consider to use TensorFlow with TensorRT which is TensorFlow module that performs optimizations.
    + Use base64 encoding when sending images.
    + Use batch predictions for large datasets.
    + Run services in same region to reduce ingress/egress charges. 

### Basic Data Exploration

Descriptive statistics: summarize information about distribution of values, central tendency, and spread of values.

Encoding data: useful for categorical data, have several types including one-hot encoding ([1, 0, 0], [0, 1, 0], [0, 0, 1]), ordinal encoding ([1, 2, 3]), binary encoding (for example binary 2 > [0, 0], [1, 0], [0, 1]), and feature hashing (convert to vector).

Feature selection: reduce the number of features to improve performance. 
+ Numeric input - numeric output:
    + Pearson's Correlation: linear relationship, ratio between covariance of two variables and product of standard deviation.
    + Spearman's Rank COefficient: non-linear relationship, based on rank value of two variables, high when variables have similar ranks.
+ Numeric input - categorical output (and vice versa):
    + ANOVA: linear relationship, difference among means.
    + Kendall's Rank Coefficient: non-linear relationship, ordinal association, rank correlation.
+ Categorical input - categorical output: 
    + Chi-Squared test: measure likelihood observed difference happened by chance.
    + Mutual information: measure of dependence between variables.

Imbalanced datasets: 
+ Occurs when distribution of instances in a training set is skewed.
+ Too few examples of some class: majority class (more frequent class) and minority class (less frequent class).
+ Minority class often more important.

Causes of imbalance:
+ Naturally
+ Biased sampling
+ Measurement errors

Mitigating impact of imbalance data:
+ Collect more data.
+ Generate synthetic data (SMOTE).
+ Resampling.
+ Use decision tree algorithms.

Feature crosses: is a type of synthetic features. 
+ Multiply (cross) two or more features to form a new feature. 
+ Usd to improve predictive capabilities of a model.
+ Especially useful with non-neural network algorithms.

Handling missing data:
+ Delete rows with missing data: completely remove rows with missing data from the datasets.
    + Advantages: removes instances that do not provide additional information.
    + Disadvantages: loss of information related to other features, not practical when significant portion of datasets has missing values for some features.
+ Replace with central tendency value: use mean, median or mode for missing numeric values.
    + Advantages: works well with small datasets.
    + Disadvantages: can increase variance and bias in dataset, can cause data leakage (e.g. time series).
+ Assign a unique category: provides a single new category NA or missing.
    + Advantages: reduces the chance of increasing variance because adding only one category.
    + Disadvantages: add another feature value to a categorical value.
+ Predict missing value: use machine learning to predict it, like regressions for numeric value, and decision tree for categorical value.
    + Advantages: unbiased relative to the rest of dataset.
    + Disadvantages: cost and time of running predicitions.
+ Last observed value carried forward: use last known value. Useful for time series. 
    + Variations: moving average and linear interpolation.

Handling outliers: instances that are significantly different from ither instances. Caused by data entry errors, measurement errors, data processing errors, sampling errors, and naturally.

Problems with outliers:
+ Increases variance.
+ Violate assumptionsof some statistical tests such regression and ANOVA.
+ Exception: outliers may be anomalies we want to detect.

Data leakage: training data contains additional information that was unintended and not available during prediciton.

Two types of data leakage:
+ Data leakage in training: like performing normalization, standardization using full datasets instead of training set only.
+ Data leakage in features: proxy variables not removed.

Ways to engineer features:
+ Transform existing feature: like map numberic variable to a scale of 0 to 1.
+ Bucketing: like creating subgroups of feature values.
+ Feature cross:
    + Cartesian product of two or more features.
    + Helps with non-linear relationships.
+ Binary features
+ Decompose values to parts
+ One-hot encoding
+ Normalization: convert numeric values to a standard. Tipically 0 to 1. 
