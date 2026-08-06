# 厂商环境锁目录

根目录 `requirements.txt` 只复现无加速卡的离线公式检查，不能用于任何芯片结论。每个
`manifest.json` 在取得真实环境前均保持 `unverified`，字段不得根据厂商名称猜测。

环境接通后，先运行 `tools/probe_environment.py` 保存探针输出，再把镜像 digest、Python、Torch、
Triton/fork、厂商扩展、驱动和编译器的精确构建号写入对应 manifest。只有这些字段均来自同一
环境且 `status` 改为 `verified`，该环境才可用于可复现的正式台账证据。

`source` 应指向平台任务、容器清单或自建环境记录；闭源组件没有包版本时，记录镜像标识、文件
hash 或厂商构建号，不能用 `>=` 版本范围替代。
