import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF 特征提取
from sklearn.neighbors import KNeighborsClassifier # KNN


dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sentence.values) # 统计词表
input_feature = vector.transform(input_sentence.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

# TF-IDF 特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
tfidf_features = tfidf_vectorizer.fit_transform(input_sentences.values)

# 配合 KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(tfidf_features, dataset[1].values)




def text_calssify_using_ml_cvt(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_ml_tfv(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = tfidf_vectorizer.transform([test_sentence])
    return knn_model.predict(test_feature)[0]

if __name__ == "__main__":
    # pandas 用来进行表格的加载和分析
    # numpy 从矩阵的角度进行加载和计算
    print("机器学习CountVectorizer: ", text_calssify_using_ml_cvt("帮我导航到天安门"))
    print("机器学习TfidfVectorizer: ", text_calssify_using_ml_tfv("儿童房的空调调高6度"))
