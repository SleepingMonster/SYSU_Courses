## 区块链2020年秋季学期期末大作业

### 成员分工

| 姓名   | 学号     | 分工     |
| ------ | -------- | -------- |
| 叶苗欣 | 18340198 | 具体实现 |
| 朱莹莹 | 18340234 | 具体实现 |
| 唐瑞怡 | 18340159 | 报告编写 |


### 应用文件account-app组织结构

```shell
|-- build.gradle // gradle配置文件
|-- gradle
|   |-- wrapper
|       |-- gradle-wrapper.jar // 用于下载Gradle的相关代码实现
|       |-- gradle-wrapper.properties // wrapper所使用的配置信息，比如gradle的版本等信息
|-- gradlew // Linux或者Unix下用于执行wrapper命令的Shell脚本
|-- gradlew.bat // Windows下用于执行wrapper命令的批处理脚本
|-- src
|   |-- main
|   |   |-- java
|   |   |     |-- org
|   |   |          |-- fisco
|   |   |                |-- bcos
|   |   |                      |-- account
|   |   |                            |-- client // 放置客户端调用类
|   |   |                                   |-- AccountClient.java
|   |   |                            |-- contract // 放置Java合约类
|   |   |                                   |-- Account.java
|   |   |-- resources
|   |        |-- conf
|   |               |-- ca.crt
|   |               |-- node.crt
|   |               |-- node.key
|   |               |-- sdk.crt
|   |               |-- sdk.key
|   |               |-- sdk.publickey
|   |        |-- applicationContext.xml // 项目配置文件
|   |        |-- contract.properties // 存储部署合约地址的文件
|   |        |-- log4j.properties // 日志配置文件
|   |        |-- contract //存放solidity约文件
|   |                |-- Account.sol
|   |                |-- Table.sol
|   |-- test
|       |-- resources // 存放代码资源文件
|           |-- conf
|                  |-- ca.crt
|                  |-- node.crt
|                  |-- node.key
|                  |-- sdk.crt
|                  |-- sdk.key
|                  |-- sdk.publickey
|           |-- applicationContext.xml // 项目配置文件
|           |-- contract.properties // 存储部署合约地址的文件
|           |-- log4j.properties // 日志配置文件
|           |-- contract //存放solidity约文件
|                   |-- Account.sol
|                   |-- Table.sol
|
|-- tool
    |-- account_run.sh // 项目运行脚本
```

### 备注
由于响应时间过长等因素，视频实在无法在2分钟内录制完成，故上述视频偏大，请见谅。
