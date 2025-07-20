# 读取 coresetlist 文件
with open('eda/docking/splits/coresetlist', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# 按字母顺序排序
lines_sorted = sorted(lines)

# 写回新文件
with open('eda/docking/splits/coresetlist.sorted', 'w', encoding='utf-8') as f:
    for line in lines_sorted:
        f.write(line + '\n')

print("排序完成，结果保存在 eda/docking/splits/coresetlist.sorted")