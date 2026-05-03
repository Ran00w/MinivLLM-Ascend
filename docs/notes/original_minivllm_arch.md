# Minivllm 整体阅读

## 调用链



## Sequence 类

### 3种状态

1. WAITTING: 等待状态
2. RUNNING： 运行状态
3. FINISHED：完成状态

### 关键功能

1. 使用block函数管理某个block的token
2. 维护sequence的各种属性，sequence是基本属性，代表一条prompt和对应的answer
3. 重写getstate和setstate两个函数，优化不同进程间通信。在decode阶段仅仅传输最后一个token，避免占用过多带宽


## Scheduler 类

### 对 prefill 和 decode 分别进行处理

对于prefill，处理的prompt数量大、消耗大量block，用waiting状态和running状态进行区分管理。decode一次只生成一个token。

prefill用allocate里面处理。decode用append处理。

### preempt 函数

用来配合踢出序列的函数。如果当前RUNNING的seq生成的token占用的block已经爆了，就直接从RUNNING最后端（最晚进入RUNNING序列）的seq直接剔除，回到WAITING状态，下次重新算

### postprocess

后处理，把生成的token_id放入seq后面

## BlockManager 类

### prefix caching 原理

使用hash把block标记。每次prefill的时候，计算出block的hash，从当前的block中找相同的。若找到就共享这一块block，若找不到就新开一个block


