"""
This example script works and shows the very basics of multiprocessing

"""


from multiprocessing import Process, Queue
import random

def rand_num(queue):
    #print("process")
    num = random.random()
    queue.put(num) # Queue is a FIFO structure. We first put stuff in, and then we can read it out
    
if __name__ == "__main__": #This is important. If it's not done this way, multiprocessing fails
                            # we cannot just run a script with multiprocessing, without using this pattern
                            # for sure not on Windows!
    

    queue = Queue() # we initiate an instance of a queue
    
    processes = [Process(target=rand_num, args=(queue,)) for x in range(5)] # we launch the processes 
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
    
    results = [queue.get() for p in processes] # we get everything back from the queue

    print("Doing stuff within Main")

    print(results)