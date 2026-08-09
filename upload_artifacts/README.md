# KernelSwift 上传产物

每个芯片目录包含 10 个可单独上传的 Python 文件：

```text
upload_artifacts/<chip_key>/<task_key>.py
```

在 KernelSwift 新建任务页中，只上传对应的 `.py` 文件；同目录的
`.manifest.json` 仅用于本地审计和证据归档，不上传。上传文件已经导出唯一的
`class Model`，并包含 `get_init_inputs()` 与 `get_inputs()`。

芯片目录使用 `profiles/chips/` 中的稳定键：

- `ascend_a2_910b`
- `metax_c500`
- `iluvatar_bi150`
- `enflame_s60`
- `thead_810e`
- `cambricon_mlu590_m9d`
- `hygon_bw1000`
- `moore_threads_ph100`
- `kunlun_p800`
- `biren_br106m`

已确认设备字面量适配的产物会在 manifest 的 `device_adaptation` 中标记为
`rewritten`；`unconfirmed_runtime` 表示尚未有足够芯片环境证据，不能把它当作
官方编译或精度通过。上传前运行：

```powershell
python tools/validate_artifacts.py
```

该命令同时检查临时生成产物和本目录中实际待上传的 100 个文件。
