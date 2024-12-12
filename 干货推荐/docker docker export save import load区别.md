## 缘起
在打包docker项目迁移部署的时候，发现了一个奇怪的现象：用docker save打包的镜像在另一台服务器上用docker import导入后，run的时候可能会出现奇怪的bash错误，而用docker load进行导入后run的时候却不会报错。本文将针对此问题进行分析，讲一下作者对docker save export load import的理解，可能对学弟学妹上手docker有一定的帮助。

## 常用命令
先来看一下四种方法常用的命令形式

1.save

下面这段代码就是将myimage：latest保存为一个名字为myimage.tar的压缩文件
```bash
docker save [OPTIONS] NAME[:TAG|@DIGEST]
docker save -o myimage.tar myimage:latest
```
2.export

下面这段代码是将mycontainer打包为mycontainer.tar，
```bash
docker export [OPTIONS] CONTAINER
docker export -o mycontainer.tar mycontainer
```

3.load

下面这段代码就是将myimage.tar加载为一个镜像，加载完成后可以用docker images进行查看
```bash
docker load -i myimage.tar
```
4.import

下面这段代码可以将mycontainer.tar加载为myrepo/myimage:latest，也可以用docker images进行查看
```bash
docker import mycontainer.tar myrepo/myimage:latest
```

## docker load和docker import的主要区别
##### 操作对象：
docker load 操作的是 Docker 镜像。

docker import 操作的是容器的文件系统快照。

##### 用途场景：
docker load 用于加载已有的 Docker 镜像。

docker import 用于从文件系统快照创建新的 Docker 镜像。

##### 结果镜像：
docker load 恢复的镜像保留了原有的所有层和元数据。

docker import 创建的镜像通常只有一层，并且没有保留原有的历史记录。

##### 文件格式：
docker load 需要的文件是由 docker save 生成的特定格式的 tar 文件。

docker import 可以接受任何包含文件系统的 tar 文件，不局限于 Docker 生成的文件。

##### 灵活性：
docker import 提供了更多的灵活性，可以通过 -c 或 --change 选项来修改导入的镜像。

docker load 则没有提供修改镜像的选项。

## 总结

通常情况下，docker save适用于镜像的备份和迁移，docker export适用于容器文件系统的备份。
总的来说，docker load 用于精确地恢复镜像，而 docker import 用于从文件系统快照创建新的镜像，并且可以对创建的镜像进行一些基本的修改。
