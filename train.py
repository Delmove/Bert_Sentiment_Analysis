import os.path
import random
import re
from datetime import datetime

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from transformers import logging
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

# # 转化为numpy数组
sentences = np.array(sentences)

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


# 对sentences数组中的每个评论字符串进行数据预处理

sentences = [preprocess(s.strip().lstrip('\ufeff')) for s in sentences]
labels = np.array(labels)


# 测试
# sentences = ['更博 爆照 帅 呀 就是 越来越 爱 生快 傻 缺爱 爱 爱',
#              '张晓鹏 jonathan   土耳其 事要 认真对待 哈哈 否则 直接 开除 丁丁 世界   细心 酒店 全部 OK 啦',
#              '姑娘 羡慕 呢 还有 招财猫 高兴 爱 蔓延 JC 哈哈 小 学徒 一枚 等 明天 见 您 呢 李欣芸 SharonLee 大佬 范儿 书呆子']
# labels = [1, 1, 1]
# print(sentences[:3])
# print(labels[:3])

# 将数据集按照5:5的数据规模随机分成训练数据与测试数据
# 将数据集分成训练集和测试集

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.5, random_state=42)

# split_idx = int(len(sentences) * 0.5)
# idx = np.arange(len(sentences))
# print(type(idx))
# np.random.shuffle(idx)
# train_sentences, test_sentences = sentences[idx[:split_idx]], sentences[idx[split_idx:]]
# train_labels, test_labels = labels[idx[:split_idx]], labels[idx[split_idx:]]


# # 自动检测情感字典的编码格式
# with open('sentiment_dict.xlsx', 'rb') as f:
#     result = chardet.detect(f.read())
#     encoding = result['encoding']

# 加载情感字典
emotion_dict = {}
df = pd.read_excel('sentiment_dict.xlsx', header=0, engine='openpyxl')
for index, row in df.iterrows():
    word = row['词语']
    emotion = row['情感分类']
    intensity = row['强度']
    polarity = row['极性']
    emotion_dict[word] = {'emotion': emotion, 'intensity': intensity, 'polarity': polarity}


#
# 测试情感字典
# emotion_dict = {
#     '逼宫': {'emotion': 'ND', 'intensity': 7, 'polarity': 0},
#     '恣纵': {'emotion': 'NN', 'intensity': 5, 'polarity': 2},
#     '唠唠叨叨': {'emotion': 'NN', 'intensity': 3, 'polarity': 2},
# }
# print(len(emotion_dict))
# 随机选择5个词语
# words = random.sample(emotion_dict.keys(), 2)
# 打印这些词语的情感信息
# for word in words:
#     emotion = emotion_dict[word]['emotion']
#     intensity = emotion_dict[word]['intensity']
#     polarity = emotion_dict[word]['polarity']
#     print(f'词语：{word}，情感分类：{emotion}，强度：{intensity}，极性：{polarity}')
#     print(f'{type(word), type(emotion), type(intensity), type(polarity)}')

# # 处理不存在于情感字典中的词
# for sentence in sentences:
#     for word in sentence:
#         if word not in emotion_dict:
#             emotion_dict[word] = {'emotion': 0, 'intensity': 0, 'polarity': 0}


class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, emotion_dict, tokenizer):
        self.sentences = sentences  # 评论列表
        self.labels = labels  # 标签列表
        self.emotion_dict = emotion_dict  # 情感字典
        self.tokenizer = tokenizer  # BERT分词器

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]  # 获取评论
        label = self.labels[idx]  # 获取标签

        # 首先，将评论进行基本的分词操作，得到一个由单词组成的列表。
        #
        # 然后，对每个单词进行处理，得到对应的ID。
        # 如果单词在BERT模型的词汇表中出现，那么就直接使用该单词在词汇表中对应的ID作为编码。
        # 如果单词不在BERT模型的词汇表中，那么就将其拆分成更小的子词，并用每个子词在词汇表中对应的ID构成一个序列，作为该单词的编码。
        # 这个过程使用了BERT模型的 subword tokenization 策略，即将单词拆分成子词来表示更多的语义信息。

        # 在得到每个单词的ID之后，BERT分词器会在序列的开头和结尾添加起始和终止token，以及在序列末尾添加填充token，使得序列长度为128。
        # 起始token（[CLS]）是用于表示序列的开始位置。
        # 终止token（[SEP]）是用于表示序列的结束位置。
        # 填充token（[PAD]）是用于填充序列长度的，以便所有序列长度相等，方便批量处理。
        # 如果序列长度小于128，则会使用填充token进行填充，使得序列长度为128。
        # 这些特殊token的ID也已经在BERT模型的词汇表中预定义好了。
        # 在使用BERT分词器进行编码时，需要将add_special_tokens参数设置为True，
        # 以便在序列的开头和结尾添加起始和终止token，并在序列末尾添加填充token。
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )  # 对评论进行编码
        return {
            'input_ids': tokenized_sentence['input_ids'].squeeze(0),  # 输入的token id
            'attention_mask': tokenized_sentence['attention_mask'].squeeze(0),  # 注意力掩码
            'label': torch.tensor(label, dtype=torch.long)  # 标签
        }
        #  'input_ids'：这个键的值是对评论进行BERT编码后得到的ID序列。
        #  BERT编码将每个词转换成对应的ID，并在序列开头和结尾添加特殊token，
        #  这样得到的ID序列可以被输入到BERT模型中进行训练和预测。
        #  该键的值是一个PyTorch tensor，形状为[seq_length]，
        #  其中seq_length是评论经过BERT编码后的长度。
        # 'attention_mask'：这个键的值是一个与'input_ids'键的值相同形状的PyTorch tensor，
        # 其中每个元素都是0或1。这个tensor被称为注意力掩码，
        # 它告诉BERT模型哪些位置是实际的词语，哪些位置是特殊token。
        # 在BERT编码的过程中，填充token的位置会被标记为0，
        # 实际词语的位置会被标记为1。
        # 'label'：这个键的值是一个表示评论情感分类标签的PyTorch tensor。
        # 在情感分类任务中，每个评论都会被分为不同的情感类别，例如积极或消极。
        # 该键的值是一个包含单个整数的tensor，表示该评论的情感类别。
        # 在这个例子中，键值对应的tensor的dtype为torch.long，
        # 这表示它是一个整数类型的tensor。



class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, emotion_dict):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model  # 加载BERT模型
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3)  # 第一层卷积，输入通道数为768，输出通道数为128，卷积核大小为3
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)  # 第二层卷积，输入通道数为128，输出通道数为64，卷积核大小为3
        self.fc = nn.Linear(in_features=64, out_features=len(emotion_dict))  # 全连接层，输入特征数为64，输出特征数为情感字典的长度
        self.emotion_dict = emotion_dict  # 情感字典

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # BERT模型的输出
        last_hidden_state = outputs.last_hidden_state  # 获取BERT模型的最后一层输出
        conv1_output = self.conv1(last_hidden_state.permute(0, 2, 1))  # 第一层卷积的输出
        conv2_output = self.conv2(conv1_output)  # 第二层卷积的输出
        pooled_output = nn.functional.max_pool1d(conv2_output, kernel_size=conv2_output.shape[2]).squeeze(2)  # 池化层的输出
        logits = self.fc(pooled_output)  # 全连接层的输出
        return logits  # 返回模型的输出结果
# 模型使用了BERT模型的最后一层输出作为输入，
# 然后通过卷积和池化操作对特征进行提取和压缩，
# 最终将提取到的特征输入到全连接层中进行分类，得到情感分类结果。

# 情感字典在模型中的作用是将模型的输出转换为对应的情感标签。
# 具体来说，模型的输出是一个向量，向量的每个维度对应着一个情感类别。
# 而情感字典则提供了每个情感类别对应的标签。
# 在模型预测时，根据输出向量的最大值所在的维度，
# 可以得到模型预测的情感类别。而情感字典则将这个维度映射为对应的情感标签，
# 从而得到最终的情感预测结果。

if __name__ == '__main__':
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', truncation=True, padding=True, max_length=128)
    # 加载数据集
    train_dataset = SentimentDataset(train_sentences, train_labels, emotion_dict, tokenizer)
    test_dataset = SentimentDataset(test_sentences, test_labels, emotion_dict, tokenizer)
    # 加载数据集迭代器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # 加载预训练模型
    bert_model = BertModel.from_pretrained('bert-base-chinese')

    # bert-base-chinese 是基于中文语料库训练的BERT模型，
    # 在该模型中使用了12层、768隐藏神经元的Transformer编码器结构。
    # 该模型包含1.04亿个参数，可对中文文本进行编码和预测，
    # 适用于各种中文自然语言处理任务，例如情感分析、命名实体识别、文本分类等。
    # BERT模型是一种预训练语言模型，可以使用大规模未标注数据进行预训练，
    # 在特定的任务上进行微调，从而实现高效的文本表示学习和语言理解能力。
    #
    # 具体来说，bert-base-chinese 模型的预训练过程包含两个阶段。
    # 首先，在大规模未标注的中文语料库上，使用MLM（Masked Language Model）和NSP（Next Sentence
    # Prediction）任务对BERT模型进行预训练。MLM任务是将输入句子中的一些单词随机遮盖，
    # 然后让BERT模型根据上下文预测遮盖单词的正确性。NSP任务是让BERT
    # 模型判断两个句子是否连续出现在文本中。在预训练完成后，可以将BERT模型用于特定任务的微调。
    #
    # bert-base-chinese 模型的性能在中文自然语言处理任务中表现优秀
    # 例如在中文情感分析任务中，通过微调 bert-base-chinese
    # 模型可以得到较好的情感分类性能。在标准的中文情感分析数据集ChnSentiCorp上，
    # 使用 bert-base-chinese 模型进行微调可以得到94.15%的准确率
    # 加载情感分类模型
    model = SentimentClassifier(bert_model, emotion_dict)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 输出训练开始信息
    print("=" * 30)
    print("开始训练".center(30))
    print("=" * 30)

    # 定义TensorBoard的SummaryWriter
    Writer_SummaryWriter = SummaryWriter()

    # 开始训练

    for epoch in range(1, EPOCHS + 1):
        logging.set_verbosity_error()
        for count, batch in enumerate(train_dataloader, 1):
            # 获取输入数据
            input_ids, attention_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']
            # 前向传播
            logits = model(input_ids, attention_mask)
            # 计算损失
            loss = criterion(logits, label)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 记录训练损失
            Writer_SummaryWriter.add_scalar('Loss/train', loss.item(), count)
            # 输出训练信息
            print(f'Epoch {epoch}.{count} loss: {loss.item()}')
            # 写入数据
            df = pd.DataFrame({'date': [NOW.strftime("%Y-%m-%d %H:%M:%S")],
                               'epoch': [epoch],
                               'count': [count],
                               'loss': [loss.item()]})
            if not os.path.isfile('loss.csv'):
                df.to_csv('loss.csv', index=False)
            else:
                df.to_csv('loss.csv', mode='a', header=False, index=False)

        # 保存模型
        name = 'model' + f'{epoch}' + '.pth'
        torch.save(model.state_dict(), name)

        # 测试模型
        with torch.no_grad():
            correct = 0
            total = 0
            true_positive = 0
            false_positive = 0
            false_negative = 0
            for batch in test_dataloader:
                # 获取输入数据
                input_ids, attention_mask, label = batch['input_ids'], batch['attention_mask'], batch['label']
                # 前向传播
                outputs = model(input_ids, attention_mask)
                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                # 统计准确率
                total += label.size(0)
                correct += (predicted == label).sum().item()
                true_positive += ((predicted == 1) & (label == 1)).sum().item()
                false_positive += ((predicted == 1) & (label == 0)).sum().item()
                false_negative += ((predicted == 0) & (label == 1)).sum().item()
            # 输出测试结果
            accuracy = 100 * correct / total
            precision = 0
            if true_positive + false_positive != 0:
                precision = true_positive / (true_positive + false_positive)
            recall = 0
            if true_positive + false_negative != 0:
                recall = true_positive / (true_positive + false_negative)
            print(f'Epoch {epoch} test accuracy: {accuracy:.2f}%, precision: {precision:.2f}, recall: {recall:.2f}')

        # 记录测试结果
        Writer_SummaryWriter.add_scalar('Accuracy/test', accuracy, epoch)
        Writer_SummaryWriter.add_scalar('Precision/test', precision, epoch)
        Writer_SummaryWriter.add_scalar('Recall/test', recall, epoch)
        # 测试结果写入文件
        NOW = datetime.now()
        dt_string = NOW.strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame({'datetime': [dt_string],
                           'epoch': [epoch],
                           'accuracy': [accuracy],
                           'precision': [precision],
                           'recall': [recall]})
        if not os.path.isfile('train_results.csv'):
            df.to_csv('train_results.csv', index=False)
        else:
            df.to_csv('train_results.csv', mode='a', header=False, index=False)

    # 关闭SummaryWriter
    Writer_SummaryWriter.flush()
    Writer_SummaryWriter.close()
