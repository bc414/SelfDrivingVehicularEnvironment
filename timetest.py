import time
delay = .5
startTime = time.time()
for i in range(1,200):
    #print(1*(time.time()-startTime-float(i)))
    #time.sleep(-1*(time.time()-startTime-float(i+1
    print(time.time()-startTime)
    print(i)
    time.sleep((i+1)*delay-time.time()+startTime)
