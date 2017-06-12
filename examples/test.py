from nlp.model.nlpmodel import NLPModel
import numpy as np

class example(NLPModel):
    """
    min 100(x[1]-x[0]^2)^2 + (1-x[0])^2

    sc   x[0] + 3x[1] = 3
         x[0]^2 + x[1]^2 <= 4
         0 <= x[0]
    """
    def __init__(self):

        NLPModel.__init__(self,2,2,'example',x0=np.array([0,1.]),
                          Lvar = np.array([0.,-np.inf]), Uvar = np.array([np.inf,np.inf]),
                          Lcon= np.array([3.0,-np.inf]), Ucon=np.array([3.0,4.]))

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
        cx[0] = x[0]**2+3.0*x[1]
        cx[1] = x[0]**2+x[1]**2
        return cx
   
    def jac(self,x):
        jac = np.zeros((2,2))
        jac[0,0] =  2.0*x[0]
        jac[0,1] =  3.0
        jac[1,0] =  2.0*x[0]
        jac[1,1] =  2.0*x[1]
        return jac

if __name__ == '__main__':
    from slp import SLP
    import logging
    from nlp.tools.logs import config_logger

    # Testing logger if debug = 1 (default=0) 

    #logger = config_logger("slp", "%(name)-3s %(levelname)-5s %(message)s")

    # Create logger.
    #slp_logger = config_logger("slp", "%(name)-8s %(levelname)-5s %(message)s",
    #                          level=logging.DEBUG,filename='slp_debug.txt',stream=None)
    
    model = example()
    model._nnln = 1
    model._nlin = 1
    model._lin = np.array([0])
    model._nln = np.array([1])

    problem = SLP(model, trustRadius=0.25,eta1=0.5, eta2=0.75,
                  trustRadiusIncrease=2., trustRadiusDecrease=0.5, verbose=1)

    problem.solve()

    print "x_opt", problem.x_k
    print "cons_dual", problem.cons_dual
    print "var_dual", problem.var_dual

