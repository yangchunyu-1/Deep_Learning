import nltk
#第一次使用 需要nltk.download('wordnet')
from nltk.corpus import wordnet

#调用wordnet.synset()方法可以得到单词的同义词簇
print(wordnet.synsets('car'))

#确认car.n.01标题词指定的同义词的含义,并获取同义词簇
car = wordnet.synset('car.n.01')
print(car.definition())
print(car.lemma_names())

#使用单词网络，查看和其他单词在语义上的上下位关系
print(car.hypernym_paths())

#求单词之间的相似度
novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')

print(car.path_similarity(novel))
print(car.path_similarity(dog))
print(car.path_similarity(motorcycle))