# FATE Pipeline

Pipeline 是一个高级 python API，允许用户按顺序设计、启动和查询 FATE 作业。FATE Pipeline的设计是用户友好的。用户可以通过向管道添加组件来自定义作业工作流，然后通过一次调用启动作业。此外，Pipeline 还提供在拟合管道后运行预测和查询信息的功能。
## FATE作业是有向无环图
FATE 作业是由算法任务节点组成的 dag。FATE流水线提供了易于使用的工具来配置任务的顺序和设置。

FATE是以模块化风格编写的。模块设计为具有输入和输出数据和模型。因此，当一个下游任务将另一个任务的输出作为输入时，会连接两个任务。通过跟踪一个数据集是如何通过FATE任务处理的，我们可以看到一个FATE作业实际上是由一系列子任务组成的。 

例如，在[官方教程中](https://github.com/FederatedAI/FATE/tree/master/doc/tutorial/pipeline_tutorial_hetero.ipynb),
guest和host的数据都是通过 `Reader`导入的.
然后`PSI` 查找guest和host之间的重复的ID. 最后, `CoordinatedLR` 的作用是拟合数据。列出的每个任务都使用数据运行一个小任务，它们共同构成了一个模型训练作业。

除了给定的教程之外，作业可能还包括多个数据集和模型。更多pipeline job示例请参考
[fate官网的示例](https://github.com/FederatedAI/FATE/tree/master/examples/pipeline)。

## 安装 Pipeline

### Pipeline CLI

成功安装FATE Client后，用户需要配置Pipeline的服务器信息。Pipeline 提供了一个用于快速设置的命令行工具。有关详细信息，请运行以下命令。

``` sourceCode bash
pipeline init --help
```

## Pipeline 的接口

### Component 组件

FATE 任务包装在 Pipeline API 中的 `component`中 。
在定义任务时，用户需要指定任务的名称、输入数据（可能命名为`input_data` 或 `train_data`), 参数，可能还需要指定输入模型。 
每个任务都可以接收和输出 `Data` 或 `Model`.
有些可能会有多份`Data` 或 `Model`. 任务参数可以在初始化时方便地设置。未指定的参数将采用默认值。所有任务都有一个 `name`，可以任意设置。任务的名称是其标识符，因此它在管道中必须是唯一的。我们建议每个任务名称都包含一个编号作为后缀，以便于跟踪。

初始化任务的示例：

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

### Data 数据

一个组件可以接收或输出多个数据。

作为一般准则，所有训练组件（即输出可重用模型的模型）都包含`train_data`, `validate_data`, `test_data`,
和 `cv_data`, 而特征工程、统计组件则包含`input_data`. 有一个例外是`Union` 组件，它接受多个输入数据。

对于输出，可以接收`train_data`, `validate_data`, `test_data`, 和 `cv_data`, 的训练组件，通常可以输出相应的输出数据。
特征工程中，统计分量通常只有`output_data`,
除了 `DataSplit` 分量, 它还有`train_output_data`, `validate_output_data`, `test_output_data`.

下面列出了所有组件的数据输入和输出：

| 算法                     | 组件名称               | 数据输入                                    | 数据输出                                                                |
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

### Model 模型

`Model` 定义组件的模型输入和输出。 与`Data`类似,
组件可以采用单个或多个输入模型。所有组件可以有一个模型输出，也可以没有模型输出。模型训练组件也可以采用 ` warm_start_model `，但请注意，应该只提供两个模型中的一个。

下面列出了所有组件的模型输入和输出：

| 算法                     | 组件名称               |  数据输入                     | 数据输出     |
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

## 构建 Pipeline

以下是构建 Pipeline 的一般指南

初始化pipeline, 应指定作业参与者和发起方。下面是pipeline的初始设置示例：
```python
from fate_client.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline().set_roles(guest='9999', host='10000', arbiter='10000')
```

用户还可以指定运行时配置：

```python
pipeline.conf.set("cores", 4)
pipeline.conf.set("task", dict(timeout=3600))
```
可以针对不同的角色单独配置所有管道任务。例如，可以专门为每一方配置任务，`Reader` 如下所示：

```python
reader_0 = Reader("reader_0", runtime_parties=dict(guest="9999", host="10000"))
reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
```

若要在管道中包含任务，请使用 `add_tasks` 。若要将`Reader`组件添加到之前创建的管道，请尝试以下操作：

```python
pipeline.add_tasks([reader_0])
```

## 运行Pipeline

添加所有组件后，用户需要先编译流水线，然后再运行设计的作业。编译完成后，Pipeline就可以拟合（运行训练作业）。
```python
pipeline.compile()
pipeline.fit()
```

## 查询任务

FATE Pipeline提供API查询任务信息，包括输出数据、模型、指标等。

```python
output_model = pipeline.get_task_info("lr_0").get_output_model()
```

## 保存 pipeline 
使用pipeline训练后，可能需要保存经过训练的pipeline，以便以后使用类似的预测

```python
pipeline.dump_model("./pipeline.pkl")
```

## 加载保存的pipeline
若要使用经过训练的pipeline进行预测，用户可以从以前保存的文件中加载它。

```python
from fate_client.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline.load_model("./pipeline.pkl")
```

## 部署组件

拟合Pipeline完成后，可以在新数据集上运行预测。在预测之前，需要首先部署必要的组件。此步骤标记要由预测管道使用的选定组件。

```python
# deploy select components
pipeline.deploy([psi_0, lr_0])
```

## 使用Pipeline进行预测

首先，启动新的Pipeline，然后指定用于预测的数据源。

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

然后，新pipeline可以启动预测

```python
predict_pipeline.predict()
```

## 加载本地文件到 DataFrame

PipeLine 提供了将本地数据表转换为 FATE DataFrame 的功能。有关快速示例，请参阅此[fate官网的演示](https://github.com/FederatedAI/FATE/tree/master/doc/tutorial/pipeline_tutorial_transform_local_file_to_dataframe.ipynb)

