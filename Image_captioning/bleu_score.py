from nltk.translate.bleu_score import sentence_bleu

x = 'I am a good boy!'
y = 'I am a boy!'
def score(x,y):
    score = sentence_bleu([x], y, weights=(1, 0, 0, 0)) 
    return score
print(score)