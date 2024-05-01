import os
from dataprocess import help as helper
def find_and_process_files(root_path, output_file):
    """
    步骤如下：
    1. 遍历给定根目录下的所有文件。
    2. 检查每个文件的文件名是否为 'modify_slice_norm.txt'，如果是，则进行下一步，否则跳过。
    3. 读取满足条件的文件的全部内容，并替换换行符。
    4. 检查文件内容的长度，如果长度大于2000或小于10，则跳过该文件。
    5. 使用 helper.rm_line_number 函数处理文件内容，去除行号。
    6. 再次检查处理后的文件内容的长度，如果长度大于1000，则跳过该文件。
    7. 将处理后的文件内容和标签 '1' 写入到输出文件中，内容和标签之间用 ' ??? ' 隔开。

    注意：这段代码假设你已经有一个名为 'helper' 的模块，并且这个模块中有一个名为 'rm_line_number' 的函数可以用来去除行号。
    """
    # 遍历给定根目录下的所有文件
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            # 检查文件是否为目标文件
            if filename == 'modify_slice_norm.txt':
                full_path = os.path.join(dirpath, filename)
                # 读取文件内容

                with open(full_path, 'r') as file:
                    # 读取全部内容并替换换行符
                    data = file.read()
                    if len(data)>2000 or len(data) < 10: # 防止补丁过大和过小
                        continue
                # 过滤行号
                    cleaned_data = helper.rm_line_numebr(data)
                if len(data)>2000 or len(data) < 10: # 防止补丁过大和过小
                    continue
                # 处理文件内容并写入到输出文件
                with open(output_file, 'a') as out_file:
                    # 格式为原始数据和标签之间用???隔开
                    out_file.write(f"{cleaned_data} ??? 1\n")

# 指定根目录路径和输出文件路径
group = ["slice1215/slice","slice20231128/slice20231128","slice20240119"]
for i in group:
    root_path = f'/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/{i}'
    output_file = f'/home/deeplearning/nas-files/SyseVR-related/data/label_data/{i.split("/")[0]}.txt'
    print("Processing files in:", root_path)
    # 调用函数
    find_and_process_files(root_path, output_file)
print("Files processed successfully!")