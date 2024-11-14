import os
import re
import matplotlib.pyplot as plt

# 用于解析每行数据的正则表达式
pattern = r'nll:.*?(\d+).(\d+).*eval_tokens:(\d+)'



# 遍历子目录中的所有txt文件
def read_txt_files_in_subdirectories(base_dir):
    # 存储所有数据
    data = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f'正在处理文件：{file_path}...')
                with open(file_path, 'r') as f:
                    for line in f:
                        match = re.match(pattern, line.strip())
                        #print(line.strip())
                        if match:
                            number1 = float(f'{match.group(1)}.{match.group(2)}')  # nll
                            number2 = int(match.group(3))    # eval_tokens
                            data.append((number2, number1))  # 将数据存入列表
                    #print(data)
            plot_data(data,file_path)
            data=[]

# 绘制图形
def plot_data(data,str):
    if not data:
        print("没有找到有效的数据！")
        return

    # 按eval_tokens进行排序
    data.sort()  # 按number2（即eval_tokens）升序排序
    print(data)
    x_vals, y_vals = zip(*data)  # 拆分数据为x和y值

    # 绘制图形
    #ax.figure(figsize=(10, 6))
    ax.plot(x_vals, y_vals, marker='o', linestyle='-', label=str)
    ax.set_xlabel('eval_tokens')
    ax.set_ylabel('nll')
    ax.set_title('nll vs eval_tokens')
    ax.grid(True)
    
fig, ax = plt.subplots()
# 主程序
def main():
    base_dir = './outputs/debug'  # 设定要遍历的根目录路径
    
    read_txt_files_in_subdirectories(base_dir)
    #plot_data()
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
