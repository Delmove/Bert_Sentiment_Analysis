import random
import re
from datetime import datetime

import jieba
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import logging

from train import SentimentClassifier, emotion_dict

logging.set_verbosity_error()

# 设置随机数种子以保证实验可重复性
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 定义超参数
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
NOW = datetime.now()

# 加载数据集
with open('weibo_senti_100k.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

# # 解析数据
sentences = []
labels = []
for line in lines:
    parts = line.strip().split(',')
    labels.append(int(parts[0]))
    sentences.append(','.join(parts[1:]))

# # 定义正则表达式，用于匹配URL链接和标点符号
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
punctuation_pattern = re.compile(r'[^\w\s]+')
#
# # 定义停用词列表
stopwords = ['的', '了', '在', '是', '我', '有',
             '和', '就', '不', '人', '都', '一',
             '一个', '上', '也', '很', '到', '说',
             '要', '去', '你', '会', '着', '没有',
             '看', '好', '自己', '这']


# # 数据预处理函数，用于对每个评论字符串进行数据清洗和预处理
def preprocess(text):
    # 删除URL链接
    text = url_pattern.sub('', text)
    # 删除标点符号
    text = punctuation_pattern.sub('', text)
    # 分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [w for w in words if w not in stopwords]
    # 返回处理后的文本，以空格分隔每个词语
    return ' '.join(words)


# # 加载训练好的模型
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
# model.load_state_dict(torch.load('model4.pth'))


def test_sentence(sentence):
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', truncation=True, padding=True, max_length=128)
    # 加载情感分类模型
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    model = SentimentClassifier(bert_model, emotion_dict)
    # 加载模型参数
    model.load_state_dict(torch.load('model4.pth'))
    # 将模型设置为评估模式
    model.eval()
    # 对输入句子进行分词
    inputs = tokenizer(sentence, return_tensors='pt')
    # 获取输入数据
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    # 获取预测结果
    _, predicted = torch.max(outputs, dim=1)
    predicted = predicted.item()  # 将tensor类型的predicted转换为int类型
    # 返回情感状态
    if predicted == 1:
        return '积极'
    elif predicted == 0:
        return '消极'


# 测试
import sys
import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

class SentimentAnalysis(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 500, 300)
        self.setWindowTitle('Sentiment Analysis By Dmd')
        self.setFixedSize(self.size()) # 设置窗口不可放缩

        # 设置背景图片
        self.background = QPixmap("background.jpg")
        self.background_label = QLabel(self)
        self.background_label.setPixmap(self.background)
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        self.background_label.setScaledContents(True)

        vbox = QVBoxLayout()

        hbox1 = QHBoxLayout()
        label = QLabel(self)
        label.setText('Enter a sentence:')
        label.setStyleSheet("QLabel { color : white; }")
        hbox1.addWidget(label)
        self.textbox = QLineEdit()
        self.textbox.setStyleSheet("QLineEdit { background-color : rgba(255, 255, 255, 150); color : black; border-radius: 10px; padding: 10px; }")
        hbox1.addWidget(self.textbox)
        vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        self.result_label = QLabel(self)
        self.result_label.setStyleSheet("QLabel { background-color : rgba(255, 255, 255, 150); color : black; border-radius: 10px; padding: 10px; }")
        hbox2.addWidget(self.result_label)
        vbox.addLayout(hbox2)

        hbox3 = QHBoxLayout()
        self.analyze_button = QPushButton('Analyze')
        self.analyze_button.setStyleSheet("QPushButton { background-color : rgba(255, 255, 255, 150); color : black; border-radius: 10px; padding: 10px; }")
        self.analyze_button.clicked.connect(self.analyze)
        hbox3.addWidget(self.analyze_button)
        vbox.addLayout(hbox3)

        # 添加时间显示
        hbox4 = QHBoxLayout()
        self.time_label = QLabel(self)
        self.time_label.setStyleSheet("QLabel { color : white; }")
        hbox4.addWidget(self.time_label)
        vbox.addLayout(hbox4)

        self.setLayout(vbox)

        # 设置定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        self.timer.start(1000)

    def analyze(self):
        sentence = self.textbox.text()
        if sentence == None:
            exit()
        self.result_label.setText(f'Sentence: {sentence}\nSentiment: {test_sentence(sentence)}\n')

        # 设置特效
        self.result_label.setAlignment(Qt.AlignCenter)

    def showTime(self):
        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.setText(current_time)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SentimentAnalysis()
    ex.show()
    sys.exit(app.exec_())




