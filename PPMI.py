import numpy as np
from corpus import preprocess,create_co_matrix
from cos_similarity import cos_similarity

#共现矩阵转化为PPMI矩阵的函数,C为共现矩阵，verbose是决定是否输出运行状况的标志
def ppmi(C,verbose=False,eps=1e-8):
    M = np.zeros_like(C,dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C,axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j] * N / (S[j] * S[i]) + eps)
            M[i,j] = max(0,pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100+1) == 0:
                    print('%.lf%% done' % (100*cnt/total))

    return M

# text = 'You say goodbye and I say hello.'
# corpus, word_to_id, id_to_word = preprocess(text)
# vocab_size = len(word_to_id)
# C = create_co_matrix(corpus,vocab_size)
# w = ppmi(C)
#
# np.set_printoptions(precision=3)
# print(C)
# print('-'*55)
# print('PPMI')
# print(w)