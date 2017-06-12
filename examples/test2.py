from nlp.model.nlpmodel import NLPModel
import numpy as np

class example(NLPModel):
    def __init__(self):

        NLPModel.__init__(self,2,2,'example',x0=np.array([6,-1]),
                          Lvar = np.array([-10.0,-np.inf]), Uvar = np.array([np.inf,3.]),
                          Lcon= np.array([0.,-np.inf]), Ucon=np.array([0.,0.]))
        self.b = np.zeros(2)
        self.b[0] = -3.0

    def obj(self,x):
        fx  = 100.0*(x[1]-x[0]**2)**2 +(1.0-x[0])**2
        return fx
    
    def grad(self,x):
        G = np.zeros(2)
        G[0] = -400.0*(x[1]-x[0]**2)*x[0]-2.0*(1.0-x[0])
        G[1] =  200.0*(x[1]-x[0]**2)
        return G

    def cons(self,x):
        cx = np.zeros(2)
        cx[0] = x[0]+3.0*x[1]#+ self.b[0]
        cx[1] = x[0]**2+x[1]**2-4.0
        return cx
   
    def jac(self,x):
        jac = np.zeros((2,2))
        jac[0,0] =  1.0
        jac[0,1] =  3.0
        jac[1,0] =  2.0*x[0]
        jac[1,1] =  2.0*x[1]
        return jac


if __name__ == '__main__':
    from slp import SLP
    
    model = example()
    #model._nlin = 1
    #model._lin = [0]
    #model._nln = [1]
    #model._nnln = 1

    algo = SLP(model, verbose=True)
    algo.solve()
    print algo.x_k
    print model.obj(algo.x_k)
    print model.cons(algo.x_k)

