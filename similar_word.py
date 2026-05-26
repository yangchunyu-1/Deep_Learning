import numpy as np
from cos_similarity import cos_similarity
from corpus import preprocess,create_co_matrix

#当某个单词被作为查询词时，将与这个单词相似的单词按降序显示出来
#query：查询词，top：显示到前几位
def most_similar(query,word_to_id,id_to_word,word_matrix,top=5):
    #取出查询词
    if query not in word_to_id:
        print('%s is not found.' % query)
        return
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    #计算余弦相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)

    #基于余弦相似度，按降序输出值
    count = 0
    for i in (-1 * similarity).argsort(): #利用负号与argsort实现降序排序
        if id_to_word[i] == query:
            continue
        print(' %s:%s' % (id_to_word[i],similarity[i]))

        count += 1
        if count > top:
            return

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus,vocab_size)
print(most_similar('you',word_to_id,id_to_word,C))