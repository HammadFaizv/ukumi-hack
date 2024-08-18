import pandas as pd
import numpy as np
from glob import glob
from os import listdir
from os.path import isfile, join, isdir
import warnings
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")


# enter your transcript and get evvaluation metrics
def evaluation(transcript : str) :
    neg="MasterDictionary/negative-words.txt"
    pos="MasterDictionary/positive-words.txt"
    stop="StopWords/"

    #MasterDictionary-20240809T095721Z-001\MasterDictionary\negative-words.txt
    print(1)
    df_1=pd.read_csv(neg,encoding="ISO-8859-1",names=["negative_words"])
    df_2=pd.read_csv(pos,encoding="ISO-8859-1",names=["positive_words"])

    def getAllFilesRecursive(root):
        files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
        dirs = [ d for d in listdir(root) if isdir(join(root,d))]
        for d in dirs:
            files_in_d = getAllFilesRecursive(join(root,d))
            if files_in_d:
                for f in files_in_d:
                    files.append(join(root,f))
        return files


    fil=getAllFilesRecursive(stop)
    # getting all stop words

    df_3=pd.read_csv(fil[0], names=['stop-words'] , encoding='ascii')
    df_4=pd.read_csv(fil[1], sep='|',names=['stop-words','country'] , encoding='latin-1')
    df_5=pd.read_csv(fil[2],encoding="ascii",names=["stop-words"]) 
    df_6=pd.read_csv(fil[3],encoding="ascii",names=["stop-words"])
    df_7=pd.read_csv(fil[4],encoding="ascii",names=["stop-words"])

    df_4.drop('country',axis=1,inplace=True)
    print(df_4.head())
    stop_words=pd.concat([df_3,df_4,df_5,df_6,df_7],axis=0)



    # creating files for all words

    stop_words_list=list(stop_words['stop-words'])
    negative_words=list(df_1['negative_words'])
    positive_words=list(df_2['positive_words'])


    print(2)
    cachedStopWords = stopwords.words("english")
    # required functions

    def func(t,cachedStopWords):
        text = t
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        return text
    def scor(t,cachedStopWords) :
        text=t
        text = [word for word in text.split() if word in cachedStopWords]
        return len(text)
    def word_count(t) :
        text=t
        text = text.split()
        return len(text)
    def sentence_count(t):
        text=t
        text = [word for word in text if word =="."]
        return len(text)
    def complex_count (t):
        text=t
        count=0
        text = text.split()
        for i in text : 
            if sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,i))) >2 :
                count=count+1
        return count
    def syllable_count (t):
        text=t
        count=0
        text = text.split()
        for i in text :
            count=count + sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,i)))
            if i[-2:] == 'es' or i[-2:]=='ed' :
                count=count-1
        return count
            

    def func_nltk(t,cachedStopWords):
        text = t
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in text:
            if ele in punc:
                text = text.replace(ele, "")

        return text
    def personal_count(t):
        text=t
        l=['I','me','Me','My','my','we',"We",'us','Us','Ours','ours']
        text = [word for word in text if word in l]
        return len(text)

    def char_count (t):
        text=t
        count=0
        text = text.split()
        for i in text :
            count=count+len(i)
        return count





    print(3)
    #enter generated transcript here
    text=transcript

    text_2=text
    # cleaning with provided stop words
    text=func(text,stop_words_list)
    count=word_count(text)
    pos=scor(text,positive_words)/count
    neg=scor(text,negative_words)/count
    #Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
    pol=(pos-neg)/((pos+neg)+0.000001)
    # (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
    subj=(pos+neg)/((count)+0.000001)
    sentence=sentence_count(text)
    if sentence == 0 :
        sentence=1
    avg_sentence=count/sentence
    complex_words=complex_count(text)
    per_complex=complex_words/count
    #Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
    fog_ind=0.4*(avg_sentence + per_complex)
    avg_word_per_sentence=avg_sentence
    # count after removing nltk stop words and  punctuations 
    cleaned_nltk=func_nltk(text,cachedStopWords)
    count_nltk=word_count(cleaned_nltk)
    count_syllable=syllable_count(text)
    syllable_per_word=count_syllable/count
    per_count=personal_count(text)
    char_counter=char_count(text)
    char_per_word=char_counter/count
    # print(char_per_word)
    result ={}
    result['POSITIVE SCORE'] = pos 
    result['NEGATIVE SCORE']=neg
    result['POLARITY SCORE']=pol
    result['SUBJECTIVITY SCORE']=subj
    result['AVG SENTENCE LENGTH']=avg_sentence
    result['PERCENTAGE OF COMPLEX WORDS']=per_complex
    result['FOG INDEX']=fog_ind
    result['AVG NUMBER OF WORDS PER SENTENCE']=avg_word_per_sentence
    result['COMPLEX WORD COUNT']=complex_words
    result['WORD COUNT']=count_nltk
    result['SYLLABLE PER WORD']=syllable_per_word
    result['PERSONAL PRONOUNS']=per_count
    result['AVG WORD LENGTH']=char_per_word
    return result

    print("done")


def main() :
    print("this is the eval file")

if __name__ == "__main__" :
    main()
    