import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def read_from_txt(filename):
    ans_dict = {}
    with open(filename, "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件
    for line in data:
        line = line.strip('\n')
        lines = line.split(" ")
        ans_dict[lines[0]] = float(lines[1])
    return ans_dict

def write_to_txt(filename, total_list):
    with open(filename, 'w+') as temp_file:
        for i1, sisdr in total_list:
            string = i1 + ": " + str(sisdr) + '\n'
            temp_file.write(string)

def takeSecond(elem):
    return elem[1]

def make_rank(sidir_list):
    sidir_list.sort(key=takeSecond, reverse=True)

def compare_two_data(data1, data2):
    ans_list = []
    for wav in data1.keys():
        num1 = data1[wav]
        num2 = data2[wav]
        ans_list.append((wav, num1 - num2))
    make_rank(ans_list)
    return ans_list
    


def draw_hist(data, filename):
    plt.hist(data, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution Histogram")
    plt.show()
    plt.savefig(filename)

def draw_two_hist(data1, data1_name, data2, data2_name, filename):
    # bins = np.linspace(5, 30, 5)
    # plt.hist(data1, bins, edgecolor="black", alpha=0.7, label=data1_name)
    # plt.hist(data2, bins, edgecolor="black", alpha=0.7, label=data2_name)
    plt.hist(data1, edgecolor="black", alpha=0.7, label=data1_name)
    plt.hist(data2, edgecolor="black", alpha=0.7, label=data2_name)
    plt.legend(loc='upper right')
    plt.xlabel("Interval(Score)")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution Histogram")
    plt.show()
    plt.savefig(filename)


data1 = read_from_txt("/workspace/project-nas-11025-sh/speech_enhance/egs/DNS-master/s1_16k/mertrics/fullsubnet_plus/NB_PESQ.txt")
data2 = read_from_txt("/workspace/project-nas-11025-sh/speech_enhance/egs/DNS-master/s1_16k/mertrics/our_fullsubnet/NB_PESQ.txt")

# print(compare_two_data(data1, data2))
write_to_txt("/workspace/project-nas-11025-sh/speech_enhance/egs/DNS-master/s1_16k/mertrics/compare/compare_NB_PESQ.txt", compare_two_data(data1, data2))