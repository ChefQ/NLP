l = "true"
def c():
    print(l)

def p():
    l = "true"
    c()
if __name__ =='__main__':
    p()