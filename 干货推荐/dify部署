## dify的部署流程，工作流开发设计

项目地址：https://github.com/langgenius/dify

部署流程：
1.安装docker和docker compose
2.从github上拉取项目下来，并进入到docker目录下
3.cp .env.example .env 复制env文件
4.docker compose up -d开始用compose.yaml脚本安装依赖，安装完成后，用docker ps确保所有容器都在运行，即可访问http://localhost/install进行使用

注意事项：可能由于版本更新的问题，compose.yaml中有部分image的tag再dockerhub上search不到，需要进行修改
dify-web:1.0.0 改为 dify-web:latest
dify-api:1.0.0 改为 dify-api:latest
还有一个是worker的image，具体是哪个忘记了，也是将1.0.0改为latest，
改完以后运行就可以拉取镜像了

拉取镜像时如果出现问题主要考虑以下几点原因：
1.镜像源有没有问题，需要挂梯子或换国内镜像源
2.是不是image的版本又更新了，需要将tag后缀改为最新版本可以使用的
