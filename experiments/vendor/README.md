# Vendor：第三方材料的本地放置规则

本目录默认只放 `README`，用于说明第三方材料的处理政策：

1. **默认不镜像**：第三方笔记、实现、PDF 只在 `resources/` 与 `experiments/` 中链接；
2. **如需离线研读**：下载到 `experiments/vendor/<name>/` 并同时补充：
   - 来源 URL 与访问日期；
   - 作者与 license（无法确认时写 `UNKNOWN`）；
   - 在 `SOURCES.yaml` 中登记；
3. **禁止**放入任何作业解答、课程官方付费材料、无许可的论文 PDF；
4. `vendor/` 中的第三方内容在 `.gitignore` 中不忽略，但推送前需逐项确认 license。

采用与 CS336 相同的 official-first + spoiler 分级原则。
