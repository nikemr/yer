import itertools  
    
count = 0
    
# # for in loop  
# a=[1,2,3,4,5,6,7,8,9,10]
# a_len=(len(a))

# while (len(a)>0):
#     print('broken')
#     # print(a)
#     for i in itertools.cycle(a): 
#         print(i, end = " ")  
#         count += 1
#         if count==10:
#             print('pop8')
#             a.pop(8)
#             break


# class Bu():
#     def __init__(self):
#         self.a=[1,2,3]
#         self.b=iter(self.a)
#         self.a_len=len(self.a)

#     def first(self):
#         if self.a_len>1: 
#             print('if')
#             self.a_len-=1    
#             print(next(self.b))
#         elif(self.a_len==1):        
#             print(next(self.b))
#             self.a_len=len(self.a)
#             print('elif')
#             self.b=iter(self.a)



class Bu():
    def __init__(self):
        self.a={"agent_1" :["Iam 1", 3],"agent_2" :["Iam 2", 12],"agent_3" :["Iam 3", 333]}
        self.b=iter(self.a.items())
        self.a_len=len(self.a)

    def first(self):
        if self.a_len>1: 
            print('if')
            self.a_len-=1    
            return next(self.b)
        elif(self.a_len==1):        
            res=next(self.b)
            self.a_len=len(self.a)
            print('elif')
            self.b=iter(self.a.items())
            return res










    



