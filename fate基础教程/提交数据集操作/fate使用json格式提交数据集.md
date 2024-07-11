## fate使用json格式提交数据集

在开始这篇文章前，你需要掌握一些linux系统使用的基本技能，包括如何使用vi编写文件并保存，文件夹的切换，文件的拷贝，文件从主机中拷贝到docker容器中等一系列操作，学习fate过程中不建议大家先系统性的学习linux系统再去做fate，可以在实践的同时自然地学会linux的使用。

使用json格式进行数据集的提交很方便，只需要遵循两个步骤即可

1. 编写json文件。
2. 使用指令进行提交。

#### 编写json文件

fate官方提供了很多的官方数据集以及对应的以及编写完成的json数据集，我们可以先从提交官方数据集开始学习

###### 1.testsuite.json格式文件

fate官方提供的数据集提交文件位置：首先找到目录examples/dsl/v2，可以看到fate所有组件的列表，点进去任意一个文件夹，都会有一个xxxx_testsuite.json格式的文件，进入后可以看到内容主要分为两大类，分别是data和tasks。在这里我们主要用到的是data这部分。

<p style="text-align: center;">
    <img alt="1" src="/photo/fate提交数据集/image-20240711161657497.png">
</p>


以截图中的文件为例，testsuite.json格式文件内每一块data都对应的一个数据集，file指向的就是数据集的位置，如红色部分中这个文件就是breast_hetero_host.csv数据集的提交文件。我们需要将红色区域指出的括号内所有内容复制下来，新建成一个数据提交文件，其中file的路径需要准确的只想到数据集所在的位置，不然可能会报错找不到文件。

###### 2.使用指令进行提交

flow data upload -c xxxx.json（刚刚编写的文件名）

提交成功后会有一个提交成功的提示，类似于下方的图片。

<p style="text-align: center;">
    <img alt="1" src="/photo/fate提交数据集/image-20240711165731211.png">
</p>

到此，你就完成了数据集的提交操作。

###### 如果你想提交自己的数据集，那么你需要学会针对数据集进行修改提交的json文件。

我在这里主要说明json文件中各个参数的含义，具体修改方法需要你自己多做尝试，熟悉参数的修改操作。

{
            "file": "examples/data/breast_hetero_host.csv",
            "head": 1,
            "partition": 4,
            "table_name": "breast_hetero_host",
            "namespace": "experiment",
            "role": "host_0"
        }

###### 参数

file: file path 

文件：文件路径

table_name & namespace: Indicators for stored data table.
表格名称 & 命名空间：存储数据表的指示器。

head: Specify whether your data file include a header or not
head：指定数据文件是否包含标题

partition: Specify how many partitions used to store the data
partition：指定用于存储数据的分区数

自己根据需要对json文件进行修改，可以完成数据集的提交操作。
