
class A:

    def __init__(self):
        print("Construct A class")
    
    def a(self):
        print("call a function")
    
    def static_fun():
        print("call static function")

    def train(self):
        pass

    def inference(self):
        pass

class B(A):

    def __init__(self):
        super().__init__()
        print("Construct B class")

    def b(self):
        print("call b function")

class C:

    def __init__(self) -> None:
        self.b = B()
        print("Construct C class")
    
    def c(self):
        print("call c function")



# a = A()

# a.a()
# A.static_fun()

# b = B()
# b.a()

c = C()
c.c()
c.b.b()
