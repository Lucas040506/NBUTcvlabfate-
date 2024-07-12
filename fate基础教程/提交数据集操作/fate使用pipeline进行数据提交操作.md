# FATE管道

Pipeline 是一个高级 python API，允许用户按顺序设计、启动和查询 FATE 作业。FATE Pipeline的设计是用户友好的。用户可以通过向管道添加组件来自定义作业工作流，然后通过一次调用启动作业。此外，Pipeline 还提供在拟合管道后运行预测和查询信息的功能。
## FATE作业是有向无环图
FATE 作业是由算法任务节点组成的 dag。FATE流水线提供了易于使用的工具来配置任务的顺序和设置。

FATE是以模块化风格编写的。模块设计为具有输入和输出数据和模型。因此，当一个下游任务将另一个任务的输出作为输入时，会连接两个任务。通过跟踪一个数据集是如何通过FATE任务处理的，我们可以看到一个FATE作业实际上是由一系列子任务组成的。 

例如，在[官方教程中](https://github.com/FederatedAI/FATE/tree/master/doc/tutorial/pipeline_tutorial_hetero.ipynb),
guest和host的数据都是通过 `Reader`导入的.
然后`PSI` 查找guest和host之间的重复的ID. 最后, `CoordinatedLR` 的作用是拟合数据。列出的每个任务都使用数据运行一个小任务，它们共同构成了一个模型训练作业。

除了给定的教程之外，作业可能还包括多个数据集和模型。更多pipeline job示例请参考
[示例](https://github.com/FederatedAI/FATE/tree/master/examples/pipeline).

## Install Pipeline

### Pipeline CLI

After successfully installed FATE Client, user needs to configure server
information for Pipeline. Pipeline provides a command
line tool for quick setup. Run the following command for more
information.

``` sourceCode bash
pipeline init --help
```

## Interface of Pipeline

### Component

FATE tasks are wrapped into `component` in Pipeline API.
When defining a task, user need to specify task's name,
input data(may be named as `input_data` or `train_data`), parameters and, possibly, input model(s).
Each task can take in and output `Data` and `Model`.
Some may take multiple copies of `Data` or `Model`. Parameters of
tasks can be set conveniently at the time of initialization.
Unspecified parameters will take default values. All tasks have a
`name`, which can be arbitrarily set. A task’s name is its
identifier, and so it must be unique within a pipeline. We suggest that
each task name includes a numbering as suffix for easy tracking.

An example of initializing a task:

```python
from fate_client.pipeline.components.fate import CoordinatedLR, PSI, Reader

lr_0 = CoordinatedLR("lr_0",
                     epochs=10,
                     batch_size=300,
                     optimizer={"method": "SGD", "optimizer_params": {"lr": 0.1}, "penalty": "l2", "alpha": 0.001},
                     init_param={"fit_intercept": True, "method": "zeros"},
                     learning_rate_scheduler={"method": "linear", "scheduler_params": {"start_factor": 0.7,
                                                                                       "total_iters": 100}},
                     train_data=psi_0.outputs["output_data"])

```

### Data

A component may take in or output multiple data input(s).

As a general guideline,
all training components(i.e. model that outputs reusable model) takes in `train_data`, `validate_data`, `test_data`,
and `cv_data`, while
feature engineering, statistical components takes in `input_data`. An exception is `Union` component,
which takes in multiple input data.

For output, training components that can take in `train_data`, `validate_data`, `test_data`, and `cv_data`, generally
may output corresponding output data. Feature engineering, statistical components usually only has `output_data`,
except for `DataSplit` component, which has `train_output_data`, `validate_output_data`, `test_output_data`.

Below lists data input and output of all components:

| Algorithm                | Component Name         | Data Input                                    | Data Output                                                                |
|--------------------------|------------------------|-----------------------------------------------|----------------------------------------------------------------------------|
| PSI                      | PSI                    | input_data                                    | output_data                                                                |
| Sampling                 | Sample                 | input_data                                    | output_data                                                                |
| Data Split               | DataSplit              | input_data                                    | train_output_data, validate_output_data, test_output_data                  |
| Feature Scale            | FeatureScale           | train_data, test_data                         | train_output_data, test_output_data                                        |
| Data Statistics          | Statistics             | input_data                                    | output_data                                                                |
| Hetero Feature Binning   | HeteroFeatureBinning   | train_data, test_data                         | train_output_data, test_output_data                                        |
| Hetero Feature Selection | HeteroFeatureSelection | train_data, test_data                         | train_output_data, test_output_data                                        |
| Coordinated-LR           | CoordinatedLR          | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas |
| Coordinated-LinR         | CoordinatedLinR        | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas |
| Homo-LR                  | HomoLR                 | train_data, validate_data, test_data, cv_data | train_output_data, validate_output_data, test_output_data, cv_output_datas |
| Homo-NN                  | HomoNN                 | train_data, validate_data, test_data          | train_output_data, test_output_data                                        |
| Hetero-NN                | HeteroNN               | train_data, validate_data, test_data          | train_output_data, test_output_data                                        |
| Hetero Secure Boosting   | HeteroSecureBoost      | train_data, validate_data, test_data, cv_data | train_output_data, test_output_data, cv_output_datas                       |
| Evaluation               | Evaluation             | input_datas                                   |                                                                            |
| Union                    | Union                  | input_datas                                   | output_data                                                                |

### Model

`Model` defines model input and output of components. Similar to `Data`,
components may take in single or multiple input models. All components can either has one or no model output.
Model training components also may take `warm_start_model`, but note that only one of the two models should be provided.

Below lists model input and output of all components:

| Algorithm                | Component Name         | Model Input                    | Model Output |
|--------------------------|------------------------|--------------------------------|--------------|
| PSI                      | PSI                    |                                |              |
| Sampling                 | Sample                 |                                |              |
| Data Split               | DataSplit              |                                |              |
| Feature Scale            | FeatureScale           | input_model                    | output_model |
| Data Statistics          | Statistics             |                                | output_model |
| Hetero Feature Binning   | HeteroFeatureBinning   | input_model                    | output_model |
| Hetero Feature Selection | HeteroFeatureSelection | input_models, input_model      | output_model |
| Coordinated-LR           | CoordinatedLR          | input_model, warm_start_model  | output_model |
| Coordinated-LinR         | CoordinatedLinR        | input_model, warm_start_model  | output_model |
| Homo-LR                  | HomoLR                 | input_model, warm_start_model  | output_model |
| Homo-NN                  | HomoNN                 | input_model, warm_start_model  | output_model |
| Hetero-NN                | HeteroNN               | input_model, warm_start_model  | output_model |
| Hetero Secure Boosting   | HeteroSecureBoost      | input_model, warm_start_model  | output_model |
| Evaluation               | Evaluation             |                                |              |
| Union                    | Union                  |                                |              |

## Build A Pipeline

Below is a general guide to building a pipeline.

Once initialized a pipeline, job participants and initiator should be
specified. Below is an example of initial setup of a pipeline:

```python
from fate_client.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline().set_roles(guest='9999', host='10000', arbiter='10000')
```

User may also specify runtime configuration:

```python
pipeline.conf.set("cores", 4)
pipeline.conf.set("task", dict(timeout=3600))
```

All pipeline tasks can be configured individually for different
roles. For instance, `Reader`
task can be configured specifically for each party like this:

```python
reader_0 = Reader("reader_0", runtime_parties=dict(guest="9999", host="10000"))
reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
```

To include tasks in a pipeline, use `add_tasks`. To add the
`Reader` component to the previously created pipeline, try
this:

```python
pipeline.add_tasks([reader_0])
```

## Run A Pipeline

Having added all components, user needs to first compile pipeline before
running the designed job. After compilation, the pipeline can then be
fit(run train job).

```python
pipeline.compile()
pipeline.fit()
```

## Query on Tasks

FATE Pipeline provides API to query task information, including
output data, model, and metrics.

```python
output_model = pipeline.get_task_info("lr_0").get_output_model()
```

## Save pipeline 
After training with pipeline, the trained pipeline may need to be saved for later using like prediction. 

```python
pipeline.dump_model("./pipeline.pkl")
```

## Load saved pipeline
To use a trained pipeline for prediction, user can load it from previous saved files.

```python
from fate_client.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline.load_model("./pipeline.pkl")
```

## Deploy Components

Once fitting pipeline completes, prediction can be run on new data set.
Before prediction, necessary components need to be first deployed. This
step marks selected components to be used by prediction pipeline.

```python
# deploy select components
pipeline.deploy([psi_0, lr_0])
```

## Predict with Pipeline

First, initiate a new pipeline, then specify data source used for
prediction.

```python
predict_pipeline = FateFlowPipeline()

deployed_pipeline = pipeline.get_deployed_pipeline()
reader_1 = Reader("reader_1", runtime_parties=dict(guest=guest, host=host))
reader_1.guest.task_parameters(namespace=f"experiment", name="breast_hetero_guest")
reader_1.hosts[0].task_parameters(namespace=f"experiment", name="breast_hetero_host")
deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]

predict_pipeline.add_tasks([reader_1, deployed_pipeline])
predict_pipeline.compile()
```

New pipeline can then initiate prediction.

```python
predict_pipeline.predict()
```

## Local File to DataFrame

PipeLine provides functionality to transform local data table into FATE DataFrame. Please refer
to
this [demo](https://github.com/FederatedAI/FATE/tree/master/doc/tutorial/pipeline_tutorial_transform_local_file_to_dataframe.ipynb)
for a quick example.
