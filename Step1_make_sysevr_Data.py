from config.Trainingconfig import TrainingConfig as Args
Args = Args()  # 类实例化
from dataprocess import normalization
import os
import threading

def process_data(file_path, output_path):
    """
    处理和正则化数据，然后将结果写入到一个新的文件中。

    参数:
    file_path (str): 输入文件的路径，该文件包含需要处理的原始数据。
    output_path (str): 输出文件的路径，处理后的数据将被写入到这个文件中。

    文件处理步骤:
    1. 读取输入文件的内容，然后使用 '------------------------------' 作为分隔符将内容分割成多个部分。
    2. 对每个部分进行处理：删除开头的空格和第一行，然后将剩余的行重新组合成字符串。
    3. 对处理后的字符串进行正则化处理，去除开头的数字。
    4. 从处理后的字符串中取出最后的数字作为标签。
    5. 将处理后的字符串和标签用 ' ??? ' 连接起来，然后添加到结果列表中。
    6. 将结果列表中的所有元素用 '\n' 连接起来，形成最终的结果字符串。
    7. 将结果字符串写入到输出文件中。

    注意: 如果在处理过程中遇到任何错误，该部分数据将被跳过。
    """
    normalized_data = []
    with open(file_path, 'r') as file:
        data = file.read().split('------------------------------')
        
        for item in data:
            try:
                item = item.strip(" ")[1:]
                item = item.split('\n')  # 将数据分割成行
                if item:  # 确保列表不为空
                    item.pop(0)  # 删除第一行
                item = '\n'.join(item)  # 将剩余的行重新组合成字符串
            except:
                continue
            data = normalization.mapping(item)  # 去除开头的数字
            # 取出最后的数字作为标签
            try:
                label = int(item.split('\n')[-2])
            except: 
                continue
            # 以???符号作为分隔符
            if isinstance(data, list): # 
                # 如果 item 是列表，则将其内部元素转换为字符串并用空格连接
                data = ' '.join(str(elem) for elem in data)
            else:
                # 如果 item 不是列表，直接转换为字符串
                data = str(data)
            normalized_data.append(data + ' ??? ' + str(label))
            
    result = '\n'.join(normalized_data)
    with open(output_path, 'w') as file:
        file.write(result)
    print(f'{os.path.basename(file_path)} processed successfully!')

def process_files_thread(root, cve_files):
    for file in cve_files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            output_path = os.path.join(root, "normal_data", file)
            process_data(file_path, output_path)

if __name__ == '__main__':
    slice_data_path = "/home/deeplearning/nas-files/SyseVR-related/data/slice20231128/"
    threads = []

    for root, dirs, cve_files in os.walk(slice_data_path):
        t = threading.Thread(target=process_files_thread, args=(root, cve_files))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    print('Data processed successfully!')
