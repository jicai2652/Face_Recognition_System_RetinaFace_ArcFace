# 人脸数据库
## 描述
    用于存放 SQLite 数据库文件，以及可能的人脸图片文件夹
***切换特征提取模型时,需要重建数据库,删除现有的数据库文件***
## 预期
**功能一**: 系统运行时自动检查或创建数据库文件，用于存储录入的人脸信息。  
**功能二**: 可能实现特征向量的维度分割存储/换用n4j等提供向量索引的数据库方案，以便检索。
> 否决功能二：跨平台部署时n4j存在环境配置问题，考虑换回SQLite
## 实现优先级:
### 功能一:
高优先级

    建立完简单界面后即应当着手建立较为简单的数据库文件,实现基本功能
### 功能二:
低优先级

    实现跨平台部署后尝试优化存储结构和检索方式