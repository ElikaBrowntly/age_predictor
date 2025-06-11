import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
import numpy as np
import sys
import os
from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModel
import torch
import requests
from io import BytesIO
import mimetypes
from typing import Union, BinaryIO

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
# 初始化jieba分词器
jieba.initialize()

# 预处理函数
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    words = jieba.cut(text)
    return ' '.join(words)

# 读取上传的文件
def read_uploaded_file(uploaded_file):
    # 获取文件名和扩展名
    file_name = uploaded_file.name.lower()
    
    # 根据文件类型使用不同的读取方法
    if file_name.endswith('.csv'):
        return pd.read_csv(uploaded_file, header=None)
    elif file_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file, header=None)
    else:
        st.error("不支持的文件格式! 请上传CSV、XLS或XLSX文件。")
        return None

# 根据文件类型选择读取方式
def read_texts_from_file(uploaded_file, sheet_name=0, column_name=None):
    # 根据文件类型选择读取方式
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding=gbk)
    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
    else:
        st.error("不支持的文件格式! 请上传CSV、XLS或XLSX文件。")
        return None
       # 自动检测列：优先使用指定列名，否则取第一列
    if column_name and column_name in df.columns:
        texts = df[column_name].tolist()
    else:
        first_column = df.columns[0]
        texts = df[first_column].tolist()
    # 确保所有元素为字符串
    return [str(text) for text in texts]

class AgeRegressionModel(torch.nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = pretrained_model
        # 冻结前3层参数
        for layer in self.bert.encoder.layer[:3]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # 回归器
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()  # 添加Sigmoid限制输出范围0-1
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 将输出缩放到0-100范围 (年龄范围)
        return self.regressor(cls_output) * 80

def predict_age(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        pred = model(input_ids, attention_mask)
    return pred.item()


# 主应用
def main():
    st.title("社交媒体年龄预测")
    
    # 模型选择
    st.subheader("选择预测场景")
    model_type = st.radio("", ("数值型年龄预测", "文本型年龄预测"), horizontal=True)
    
    # 文件上传 - 支持CSV和Excel格式
    st.subheader("上传数据文件")
    uploaded_file = st.file_uploader(
        "选择CSV或Excel文件", 
        type=['csv', 'xls', 'xlsx'],
        help="支持CSV、XLS和XLSX格式的文件"
    )
    
    st.markdown(f"""
    **支持XLS、XLSX、CSV格式**
    """)
    
    if uploaded_file is not None:
        try:
            # 显示文件信息
            st.info(f"已上传文件: {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)")
            
            # 读取文件
            with st.spinner('正在读取文件...'):
                data = read_uploaded_file(uploaded_file)
                
            if data is None:
                return
                
            st.info(f"成功读取文件，共 {len(data)} 条记录")
            
            if model_type=="数值型年龄预测":
            
                # 跳过第一行（标题行）
                if len(data) > 0:
                    data = data.iloc[1:]
                    st.info(f"已跳过标题行，剩余 {len(data)} 条数据记录")
                else:
                    st.error("文件为空或格式错误!")
                    return
                
                data = data.replace(["female"], [0]).replace(["male"], [1])
            
                if len(data) == 0:
                    st.error("没有有效数据可供预测")
                    return
                                        
                # 加载模型
                model_file = r"C:\Users\Lenovo\Desktop\信息对抗\default_pred.joblib"
                if not os.path.exists(model_file):
                    st.error(f"模型文件 {model_file} 不存在")
                    return
                
                with st.spinner('正在加载模型并预测...'):
                    # 加载模型
                    model = joblib.load(model_file)
                
                    data = data.drop(columns=[11])
                    TestData = np.array(data.iloc[0:, :]).tolist() 
                
                    # 预测年龄
                    predictions = model.predict(TestData)
                
                    # 显示结果
                    results = pd.DataFrame({
                        '预测年龄': predictions
                    })
                    st.dataframe(results)
                
            else:
                test_texts = read_texts_from_file(
                uploaded_file,
                sheet_name="Sheet1",    # Excel工作表名
                column_name="content"      # 指定列名（可选）
                )
                # 验证结果
                print("读取的文本列表:")
                for i, text in enumerate(test_texts, 1):
                    print(f"{i}. {text}")
                    
                model=joblib.load(r"C:\Users\Lenovo\Desktop\信息对抗\age_regression_model.joblib")
                    
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_path = r"C:\Users\Lenovo\Desktop\hflchinese-roberta-wwm-ext"
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                # 初始化存储预测结果的列表
                predictions = []

                # 对每个文本进行预测并存储结果
                for text in test_texts:
                    predicted_age = predict_age(text, model, tokenizer, device)
    
                    # 打印到控制台（可选）
                    print(f"文本: '{text[:30]}{'...' if len(text) > 30 else ''}' | 预测年龄: {predicted_age:.1f}岁")
    
                    # 将预测结果添加到列表中
                    predictions.append({
                        '原始文本': text,  # 存储完整文本
                        '预测年龄': f"{predicted_age:.1f}岁"  # 格式化年龄
                    })

                # 创建DataFrame显示结果
                if predictions:
                    results = pd.DataFrame(predictions)
    
                    # 显示完整结果表格
                    st.dataframe(results)

            st.success(f"{model_type}完成!")

        except Exception as e:
            st.error(f"处理错误: {str(e)}")
            st.error(f"错误详情: {sys.exc_info()[0]}")

if __name__ == "__main__":
    main()