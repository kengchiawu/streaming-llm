import os
import re
import matplotlib.pyplot as plt

# 用于解析每行数据的正则表达式
pattern = r'nll:.*?(\d+).(\d+).*eval_tokens:(\d+)'

pattern_name = r'.*log_ppl_.*/(\w+).txt' 
file1 = 'C:/Users/y1116/Documents/GitHub/streaming-llm/outputs/debug/log_ppl_Attention_Sink.txt'
match_test = re.search(pattern_name, file1)
if match_test:
    print(match_test.group(1))
else:
    print("no match")

# 遍历子目录中的所有txt文件
def read_txt_files_in_subdirectories(base_dir):
    # 存储所有数据
    data = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_name, _ = os.path.splitext(os.path.basename(file_path))
                #print(file_name)
                with open(file_path, 'r') as f:
                    for line in f:
                        match = re.match(pattern, line.strip())
                        match_name = re.search(r'log_ppl_(\w+)', file_name)
                        #print(line.strip())
                        if match:
                            number1 = float(f'{match.group(1)}.{match.group(2)}')  # nll
                            number2 = int(match.group(3))    # eval_tokens
                            data.append((number2, number1))  # 将数据存入列表
                    #print(data)
            if match_name:
                plot_data(data,match_name.group(1))
            data=[]

# 绘制图形
def plot_data(data,str):
    if not data:
        print("没有找到有效的数据！")
        return

    # 按eval_tokens进行排序
    data.sort()  # 按number2（即eval_tokens）升序排序
    x_vals, y_vals = zip(*data)  # 拆分数据为x和y值

    # 绘制图形
    #ax.figure(figsize=(10, 6))
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=str)
    ax.set_xlabel('eval_tokens')
    ax.set_ylabel('log PPL')
    ax.set_title('llama 7B')
    ax.grid(True)
    
fig, ax = plt.subplots()
# 主程序
def main():
    base_dir = './outputs/debug'  # 设定要遍历的根目录路径
    
    read_txt_files_in_subdirectories(base_dir)
    #plot_data()
    ax.axvline(x=1000, color='r', linestyle='--', label='KV Cache Size')  # 第一条竖虚线
    ax.axvline(x=2200, color='g', linestyle='--', label='Pre-training Length')  # 第二条竖虚线
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
