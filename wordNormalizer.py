
class StaticArray:
    bn_norm={
        '09C0':'\u09bf','09C2':'\u09c1','09C4':'\u09c3','09A3':'\u09a8','0988':'\u0987','098A':'\u0989'
    }

def word_normalize(word):
        try:
            s = ""
            for c in word:
                g = c.encode("unicode_escape")
                g = g.upper()
                g = g[2:]
                g = g.decode('utf-8')
                if g in StaticArray.bn_norm:
                    g = StaticArray.bn_norm[g].encode().decode('utf-8')
                    s+=g
                    continue
                s+=c
            return s
        except:
            print("ERROR 204: Error in word normalization!!")
            return word
        
if __name__ == "__main__":
    word = "কুয়াকাটা"
    word2 = "কুয়াকাটা"
    import codecs
    print(word.encode("unicode_escape"))
    encoded_word = codecs.encode(word, 'unicode_escape')
    pattern = r"য়" # double sequence of unicode
    import re 
    seq = re.findall(pattern, word)
    print(seq)
    seq = re.findall(pattern, word2)
    print(seq)