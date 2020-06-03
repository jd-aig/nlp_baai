import re
import jieba

pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
pattern_css = re.compile(r'<.+>')
pattern_pun = '！，；：？、。"!,;:?."\''


def tokenize_spt(text):

    sp_token = ['<img>', '<url>', '<sos>', '<eos>']

    resp_list = list()
    tmp_list = jieba.cut(text, cut_all=False)

    seg_list = list(tmp_list)
    i = 0

    while i < len(seg_list):
        if ''.join(seg_list[i:i + 3]) in sp_token:
            resp_list.append(''.join(seg_list[i:i + 3]))
            i = i + 3
        else:
            resp_list.append(''.join(seg_list[i]))
            i = i + 1

    return resp_list


def clean_css(text):
    url_list = re.findall(pattern_css, text)
    for url in url_list:
        text = text.replace(url, '')

    return text


def clean_sentence(text):

    text = text.replace('未购买→售前咨询组', '')
    text = text.replace('已购买→售后咨询组', '')
    text = text.replace(' ||| ', '')

    img = "NULL"
    url_list = re.findall(pattern, text)
    for url in url_list:
        text = text.replace(url, '<url>')

    return text


def clean_punctuation(text):
    text = re.sub(r'[{}]+'.format(pattern_pun), '', text)
    return text.strip().lower()


def clean_text(text):
    text = clean_css(text)
    text = clean_sentence(text)
    text = clean_punctuation(text)
    text = ' '.join(tokenize_spt(text))
    return text


def process(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    data_size = len(lines)
    img = ''
    session = ''
    last_sid = lines[0].strip().split('\t')[0]
    last_text = lines[0].strip().split('\t')[3]

    with open('./data/'+path.split('_')[-1], 'w') as f:
        for i in range(1,data_size):
            line = lines[i]

            line = line.strip()
            line = line.split('\t')

            sid = line[0]
            text = line[3]

            if i == data_size-1:
                #读完了，全部写入
                if last_text[-4:] == '.jpg':
                    img += last_text + ' '
                    session += '<img>'
                else:
                    img += 'NULL' + ' '
                    session += clean_text(last_text)

                if text[-4:] == '.jpg':
                    img += text
                    session += '\t' + '<img>' + '\t'
                else:
                    img += 'NULL'
                    session += '\t' + clean_text(text) + '\t'
                f.write(session+img+'\n')

            elif last_sid != sid:
                #开启新对话，把last作为target写入
                if last_text[-4:] == '.jpg':
                    session = session[:-4] + '\t' + '<img>' + '\t'
                else:
                    session = session[:-4] + '\t' + clean_text(last_text) + '\t'
                f.write(session + img +'\n')
                session = ''
                img = ''

            else:
                #未开启新对话，录入last
                if last_text[-4:] == '.jpg':
                    img += last_text + ' '
                    session += '<img>' + '</s>'
                else:
                    img += 'NULL' + ' '
                    session += clean_text(last_text) + '</s>'
            last_sid = sid
            last_text = text
      

if __name__ == '__main__':
    
    process('./data/data_dev.txt')
    print('dev done!')
    process('./data/data_train.txt')
    print('train done!')