def dosya():
    file = open("yesno.txt","r", encoding="utf-8")#türkçe olması için utf-8
    lines = file.readlines()
    file.close

    countYes = 0
    countNo = 0
    for line in lines:
        line = line.strip().upper()
        
        #print(line)
        if(line.find("YES")>=0)and len(line)==3 :
            countYes+=1
        if(line.find("NO")>=0):
            countNo+=1
    print("yes:",countYes," no:",countNo)
dosya()