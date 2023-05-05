# 北京交通大学课程实验

## 大数据实验

华为云华为大数据平台搭建

本实验基于华为云OBS和华为云ECS服务构建一个存算分离的基本架构，并通过运行一个计算程序来完成存算分离架构的验证。本实验的实验数据存储在OBS中，通过在ECS上部署开源组件（Hadoop和Spark）构成计算环境，最后编写Spark程序访问存储在OBS上的数据进行计算（单词出现次数统计）并输出结果。

## 操作系统实验

本实验旨在一步一步展示如何从零开始用 Rust 语言写一个基于 RISC-V 架构的类 Unix 内核 。实验内容主要分为以下几个方面。

- [Lab 0](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 配置操作系统开发的基本环境。
- [Lab 1](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 构建一个独立的不依赖于 Rust 标准库的可执行程序。
- [Lab 2](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现裸机上的执行环境以及一个最小化的操作系统内核。
- [Lab 3](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现一个简单的批处理操作系统并理解特权级的概念。
- [Lab 4](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现一个支持多道程序和协作式调度的操作系统。
- [Lab 5](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现一个分时多任务和抢占式调度的操作系统。
- [Lab 6](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现内存的动态申请和释放管理。
- [Lab 7](https://github.com/F1reC/labs/tree/main/%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F) - 实现进程及进程的管理。

共计 8 个实验项目，通过实验的方式深入研讨操作系统底层的工作原理，并使用 Rust 语言逐步实现一个基本的操作系统内核。


## 机器学习实验

实现了四个经典的kaggle项目：

- 波士顿房价预测
- 恶性乳腺癌肿瘤预测
- Spaceship Titanic预测
- 鸢尾花分类



