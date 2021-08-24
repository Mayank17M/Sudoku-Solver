def sudoku(f):

    def af(g):
        for n,l in enumerate(g):
            for m,c in enumerate(l):
                print(str(c).replace("0","."),end="")
                if m in {2,5}:
                    print("+",end="")
            print()
            if n in {2,5}:
                print("+",*11)
    
    def cp(q,s):
        l=set(s[q[0]])
        l |= {s[i][q[1]] for i in range(9)}
        k = q[0]//3, q[1]//3
        for i in range(3):
            l |= set(s[k[0]*3 +i][k[1]* 3:(k[1]+1)*3])
        return set(range(1,10))-l

    def ec(l):
        q=set(l)-{0}
        for c in q:
            if l.count(c)!=1:
                return True
        return False

    af(f)

    s= []
    t= []
    for nl,l in enumerate(f):
        try:
            n=list(map(int,l))
        except:
            print("Line "+ str(nl+1) + "contains something other than a number.")
            return
        if len(n) != 9:
            print("Line " + str(nl + 1) + " does not contain 9 digits.")
            return
        t += [[nl, i] for i in range(9) if n[i] == 0]
        s.append(n)

    if nl != 8:
        print("Sudoku contains: "+ str(nl+1) + "lines instead of 9.")
        return
    
