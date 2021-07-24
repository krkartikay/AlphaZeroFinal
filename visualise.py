# define visualisation of board

def vis(x):
    r = [('-1','O'),('1','X'),('0','.')]

    for k, v in r:
        x = x.replace(k, v)

    i = 0
    s = ""
    for c in x:
        if c in 'OX.':
            i += 1
            if i % 3 == 0:
                s += c + "\n"
            else:
                s += c + " "
    
    return s

if __name__ == '__main__':
    while True:
        x = input("Input: ")        
        print(vis(x))