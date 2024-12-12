### 缘起
在打包docker项目迁移部署的时候，发现了一个奇怪的现象：用docker save打包的镜像在另一台服务器上用docker import导入后，run的时候可能会出现奇怪的bash错误，而用docker load进行导入后run的时候却不会报错。本文将针对此问题进行分析，讲一下作者对docker save export load import的理解，可能对学弟学妹上手docker有一定的帮助。

## 常用命令
先来看一下四种方法常用的命令形式
save
下面这段代码就是将myimage：latest保存为一个名字为myimage.tar的压缩文件
'''bash
docker save [OPTIONS] NAME[:TAG|@DIGEST]
docker save -o myimage.tar myimage:latest
'''
