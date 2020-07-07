#class
class Man:
    def __init__(self,name): #생성자
        self.name=name
        print("Initialized!")

    def hello(self):
        print("Hello"+self.name+"!")

    def goodbye(self):
        print("Good-bye"+self.name+"!")

m=Man("David")
m.hello()
m.goodbye()