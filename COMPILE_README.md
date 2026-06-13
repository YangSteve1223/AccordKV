# ACCORD-KV 论文编译说明

## 环境要求
- XeLaTeX (测试于 TeXLive 2024)

## 编译步骤
```bash
cd /root/accord-kv
bash compile_paper.sh
```

## 文件说明
- `ACCORD_KV_paper.tex` - 论文源文件
- `ACCORD_KV_paper.bbl` - 参考文献 (BibTeX)
- `compile_paper.sh` - 编译脚本

## 注意事项
1. 需要两次 XeLaTeX 运行以正确解析交叉引用
2. 编译后会在当前目录生成 `ACCORD_KV_paper.pdf`
