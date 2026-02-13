# 候选图搜索几何求解评测报告

- 样本数: 15
- Top-1 准确率: 0.800
- Top-3 召回率: 1.000
- unknown 精确率: 0.667

## 失败样例
- `shell_en_1`: expected=shell, predicted=unknown, ranked=[('single_tubs', 0.2824180362765086), ('single_sphere', 0.2824180362765086), ('shell', 0.26530719245228224)]
- `grid_zh_1`: expected=grid, predicted=single_box, ranked=[('single_box', 0.4595740523754843), ('grid', 0.31983335272444485), ('ring', 0.11220191799908037)]
- `nest_zh_1`: expected=nest, predicted=single_tubs, ranked=[('single_tubs', 0.3658256416344335), ('nest', 0.26564393740970904), ('single_sphere', 0.26564393740970904)]
