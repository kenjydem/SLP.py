from cylp.cy import CyClpSimplex, CyCoinPackedMatrix
from nlp.tools.dercheck import DerivativeChecker
from nlp.model.nlpmodel import LPModel
from scipy.sparse import coo_matrix, csc_matrix
from scipy.optimize import fsolve
from numpy.linalg import norm
import numpy as np
import logging, copy, warnings

class SLP(object):
    """"Sequential Linear Programming solver."""

    def __init__(self, model, **kwargs):

        """Initialize a SLP solver for a NLP model with `n` variables and `m` constraints.
        :parameters:
            :model:                   a class 'NLPModel' instance

        :keywords:

            :eta1:                   first prediction's state                                   (default: 1.0e-1)
            :eta2:                   second prediction's state                                  (default: 7.0e-1)

            :mu:                     penalty parameters for the measure of infeasability        (default: 1.0e+3)
            :minMu:                  maximum value of mu                                        (default: 1.0e+4)
            :maxMu:                  maximum value of mu                                        (default: 1.0e-1)

            :trustRadius:            initial radius of trust-region                             (default: all 5.)
            :trustRadiusIncrease:    coefficient of trust-region increasement                   (default: 10.)
            :trustRadiusDecrease:    coefficient of trust-region decreasement                   (default: 1./3)

            :major_iter:             maximum number of major iterations in SLP                  (default: max(1000,m))
            :minor_iter:             maximum number of minor iterations in LP solver            (default: max(500,3*m))
            :atol:                   stopping tolerance for Relative objective change (Rel_obj) (default: 1.0e-6)
            :nIterNoProgress:        maximum of iteration without satisfied Rel_obj<atol        (default: 10)
            :step_tol:               stopping tolerance for step norm                           (default: 1.0e-6)
            :opt_tol:                stopping tolerance for KKT                                 (default: 1.0-3)
            :feas_tol:              stopping tolerance for feasability error                    (default: 1.0-5)

            :verbose:                verbose solver                                             (default: False)
            :debug:                  Using logger                                               (default: False)

            :backtracking:           perform Armijio backtracting if step no accpeted           (default: False)
            :backtrackingSegments:   number of iteration during backtracking                    (default: 5)

        """

        self.model = model

        self.eta1 = kwargs.get("eta1", 5.0e-1)
        self.eta2 = kwargs.get("eta2", 7.0e-1)

        self.mu = kwargs.get("mu", 1.)
        self.stored_mu_init = copy.copy(kwargs.get("mu", 1.))
        self.maxMu = kwargs.get("maxMu", 1.0e+3)

        self.trustRadius = kwargs.get("trustRadius", 10.)
        self.stored_trustradius_init = copy.copy(kwargs.get("trustRadius", 10.))
        self.trustRadiusIncrease = kwargs.get("trustRadiusIncrease", 3.)
        self.trustRadiusDecrease = kwargs.get("trustRadiusDecrease", 1./3)

        self.major_iter = kwargs.get("major_iter", max(500,model.m))
        self.minor_iter = kwargs.get("minor_iter", max(500,3*model.m))

        self.feas_tol = kwargs.get("feas_tol", 1.0e-6)       # Sum of violation at the optimality 
        self.feas_accept = kwargs.get("feas_accept", 10.0)   # Sum of violation accpetable during process
        self.step_tol = kwargs.get("step_tol", 1.0e-5)       # tolerance of the step size 
                                    
        self.relChange_tol = kwargs.get("relChange_tol", 1.0e-6)
        self.nIterNoProgress = kwargs.get("nIterNoProgress", 100)

        self.verbose = kwargs.get("verbose", False)
        self.debug = kwargs.get("debug", False)

        self.backtracking = kwargs.get("backtracking", False)
        self.backtrackingSegments = kwargs.get("backtrackingSegments", 5)

        self.p_ind = np.intersect1d(model.nln, model.upperC)
        self.np_ind = len(self.p_ind)
        self.q_ind = np.intersect1d(model.nln, model.lowerC)
        self.nq_ind = len(self.q_ind)
        self.r_ind = np.intersect1d(model.nln, model.rangeC)
        self.nr_ind = len(self.r_ind)
        self.z_ind = np.intersect1d(model.nln, model.equalC)
        self.nz_ind = len(self.z_ind)

        self.nnlc = self.np_ind+ self.nq_ind+self.nr_ind + self.nz_ind   # Number of elastic variables
        self.nlin = model.nlin                                           # Number of linear constraints

        self.header_fmt = '%3s  %12s  %7s %7s  %7s  %7s  %8s  %8s  %8s  %7s  %7s %7s %7s %7s'
        self.fmt = '%3d  %12.5e  %7.1e %7.1e  %7.1e  %7.1e  %8.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e %7.1e  %7.1e'

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get("logger_name", "slp")
        self.log = logging.getLogger(logger_name)
        self.log.propagate = False
        warnings.filterwarnings("ignore")

        self.resetStoredValues()
        self.x_k = model.x0
        self.initializeSubproblem()
        self.initializeSolver()


    def initializeSubproblem(self):
        """
        Initialize linearized subproblem

        We construct the problem such as the matrix constraints is


                                n     p    q   r   z  
        [-Lcon]               [  J    0    0   0   0 ] [x]    [Ucon]              } nlin
        [-infty]              [  J   -I    0   0   0 ] [p]    [Ucon -c(x)+J*x_k]  } np 
        [-infty]              [  -J   0   -I   0   0 ] [q]    [c(x) -Lcon- J*x_k] } nq 
        [-infty]           <= [  J    0    0   -I  0 ] [r] <= [Ucon -c(x)+J*x_k]  } nr 
        [-infty]              [  -J   0    0   -I  0 ] [z]    [c(x) -Lcon- J*x_k] } nr
        [Lcon + J*x_k -h(x)]  [  J    0    0   0   I]         [infty]             } nz 
        [- infty]             [  J    0    0   0  -I]         [Ucon + J*x_k -h(x)]} nz 
        """

        model = self.model

        # Checking nonlinear constraints
        if model.nnln > 0:

            # Initialize coefficient objective
            pq_coefs = self.mu * np.ones(self.nnlc)
            c = np.concatenate((self.gFxk,pq_coefs))
            J_x_k = self.J_x_k

            # Initialize upper bound
            lower_bounds= np.concatenate((model.Lvar,np.zeros(self.nnlc)))

            # Initialize upper bound
            upper_bounds = np.concatenate((model.Uvar,np.ones(self.nnlc)*np.inf))

            # Initialize sub-matrix constraint
            nonlinear_linearized_upper_coef = nonlinear_linearized_lower_coef = \
                nonlinear_linearized_range_coef = nonlinear_linearized_equal_coef = \
                linear_constraints = np.empty((0,model.n + self.nnlc))

            if self.np_ind > 0:
                # Initialize sub-matrix constraints for non linear upper constraints
                nonlinear_linearized_upper_coef = np.concatenate((J_x_k[self.p_ind,:],
                                                                  -np.eye(self.np_ind),
                                                                  np.zeros((self.np_ind, self.nq_ind+self.nr_ind + self.nz_ind))), axis=1)


            if self.nq_ind > 0:
                # Initialize sub-matrix of constraints for non linear lower constraints
                nonlinear_linearized_lower_coef = np.concatenate((-J_x_k[self.q_ind,:],
                                                                  np.zeros((self.nq_ind, self.np_ind)),
                                                                  -np.eye(self.nq_ind),
                                                                  np.zeros((self.nq_ind, self.nr_ind + self.nz_ind))), axis=1)


            if self.nr_ind > 0:
                # Initialize sub-matrix of constraints for non linear range constraints
                nonlinear_linearized_range_coef = np.concatenate((np.concatenate((J_x_k[self.r_ind,:],
                                                                  np.zeros((self.nr_ind, self.np_ind+self.nq_ind)),
                                                                  -np.eye(self.nr_ind),
                                                                  np.zeros((self.nr_ind, self.nz_ind))), axis=1),

                                                                  np.concatenate((-J_x_k[self.r_ind,:],
                                                                  np.zeros((self.nr_ind, self.np_ind+self.nq_ind)),
                                                                  -np.eye(self.nr_ind),
                                                                  np.zeros((self.nr_ind, self.nz_ind))), axis=1)))

            if self.nz_ind > 0:
                # Initialize sub-matrix of constraints for non linear equality constraints
                nonlinear_linearized_equal_coef = np.concatenate((np.concatenate((J_x_k[self.z_ind,:],
                                                                  np.zeros((self.nz_ind, self.np_ind+self.nq_ind+self.nr_ind)),
                                                                  np.eye(self.nz_ind)), axis=1),

                                                                  np.concatenate((J_x_k[self.z_ind,:],
                                                                  np.zeros((self.nz_ind, self.np_ind+self.nq_ind+self.nr_ind)),
                                                                  -np.eye(self.nz_ind)), axis=1)))

            if model.nlin >0:
                # Initialize sub-matrix for linear constraints
                linear_constraints = np.concatenate((self.J_L,np.zeros((model.nlin, self.nnlc))),axis=1)
                linear_comp = np.dot(J_x_k[model.lin,:], self.x_k) - self.cons_xk[model.lin] 

            A = np.concatenate((linear_constraints,
                                nonlinear_linearized_upper_coef,
                                nonlinear_linearized_lower_coef,
                                nonlinear_linearized_range_coef,
                                nonlinear_linearized_equal_coef))

            Lcon_lin = np.concatenate((model.Lcon[model.lin] + linear_comp, 
                                        -np.ones(self.np_ind) * np.inf,
                                        -np.ones(self.nq_ind) * np.inf,
                                        -np.ones(2*self.nr_ind) * np.inf,
                                        np.zeros(self.nz_ind),
                                        -np.ones(self.nz_ind) * np.inf))

            Ucon_lin = np.concatenate((model.Ucon[model.lin] + linear_comp,
                                      np.zeros(self.np_ind+self.nq_ind+2*self.nr_ind),
                                      np.ones(self.nz_ind) * np.inf,
                                      np.zeros(self.nz_ind)))

        else:
            # Problem has only linear constraint
            A = self.J_x_k
            c = self.gFxk
            linear_comp = np.dot(A, self.x_k) - self.cons_xk
            Lcon_lin = model.Lcon + linear_comp  
            Ucon_lin = model.Ucon + linear_comp 
            lower_bounds = model.Lvar.copy()
            upper_bounds = model.Uvar.copy()


        self.LinearModel = LPModel(c.copy(), A, name = 'Linearized_model',
                                   Lcon = Lcon_lin, Ucon = Ucon_lin,
                                   Lvar = lower_bounds, Uvar = upper_bounds)

        self.LinearModel.Asparse = coo_matrix(A)

        return

    def initializeSolver(self):
        """
        Initialize linear solver to compute the step and the projection
        """

        self.LP_solver = CyClpSimplex()
        self.LP_solver.logLevel = 0
        self.LP_solver.maxNumIteration = self.minor_iter

        A = CyCoinPackedMatrix(colOrdered=True,
                                rowIndices=self.LinearModel.Asparse.row,
                                colIndices=self.LinearModel.Asparse.col,
                                elements=self.LinearModel.Asparse.data)

        self.LP_solver.loadProblem(A, self.LinearModel.Lvar,
                                   self.LinearModel.Uvar,
                                   self.LinearModel.c,
                                   self.LinearModel.Lcon,
                                   self.LinearModel.Ucon)

        # Initialisation de la projection
        self.proj = CyClpSimplex()
        self.proj.logLevel = 0

        ci = np.concatenate((np.ones(self.model.n), np.zeros(self.model.n)))
        I = np.eye(self.model.n)

        if self.nlin > 0:
            A = np.vstack((np.hstack((-I,I)),
                           np.hstack((-I,-I)),
                           np.hstack((np.zeros(self.J_L.shape),self.J_L))))

            Lcon = np.hstack((-np.inf*np.ones(2*self.model.n), self.model.Lcon[self.model.lin]))
            Ucon = np.hstack((self.x_k, -self.x_k, self.model.Ucon[self.model.lin]))
        else:
            A = np.vstack((np.hstack((-I,I)),
                           np.hstack((-I,-I))))

            Lcon = -np.inf*np.ones(2*self.model.n)
            Ucon = np.hstack((self.x_k, -self.x_k))

        Asparse = coo_matrix(A)

        Ai = CyCoinPackedMatrix(colOrdered=True,
                                rowIndices=Asparse.row,
                                colIndices=Asparse.col,
                                elements=Asparse.data)

        self.proj.loadProblem(Ai, np.hstack((np.zeros(self.model.n),self.model.Lvar)),
                                  np.hstack((np.inf*np.ones(self.model.n),self.model.Uvar)), ci, Lcon, Ucon)



    def updateSubproblem(self):
        """
        Update linearized subproblem.
        Only nonlinear-constraints are updated.
        """
        model = self.model
        x_k = self.x_k
 
        if model.nnln > 0:
            # Update objective subproblem
            pq_coefs = self.mu * np.ones(self.nnlc)
            self.LinearModel.c = np.concatenate((self.gFxk, pq_coefs))

            if self.np_ind > 0:
                J_x_k = self.J_x_k[self.p_ind]

                # Update constraint Matrix
                self.LinearModel.A[self.nlin:self.nlin+self.np_ind,:model.n] =  J_x_k
                # Update nonlinear linearized right hand side
                self.LinearModel.Ucon[self.nlin : self.nlin + self.np_ind] = \
                            -self.cons_xk[self.p_ind] + np.dot(J_x_k,x_k) + model.Ucon[self.p_ind]

            if self.nq_ind > 0:
                J_x_k = self.J_x_k[self.q_ind]

                # Update constraint Matrix
                self.LinearModel.A[self.nlin+self.np_ind:self.nlin+self.np_ind+self.nq_ind,:model.n] = -J_x_k
                # Update nonlinear linearized right hand side
                self.LinearModel.Ucon[self.nlin+self.np_ind:self.nlin+self.np_ind+self.nq_ind] = \
                            self.cons_xk[self.q_ind] - np.dot(J_x_k,x_k) - model.Lcon[self.q_ind]


            if self.nr_ind > 0:
                J_x_k = self.J_x_k[self.r_ind]

                # Update constraint Matrix                 
                self.LinearModel.A[self.nlin+self.np_ind+self.nq_ind:self.nlin+self.np_ind+self.nq_ind+self.nr_ind,
                               :model.n] = J_x_k
                               
                self.LinearModel.A[self.nlin+self.np_ind+self.nq_ind+self.nr_ind:self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind,
                               :model.n] = -J_x_k
                               
                # Update nonlinear linearized right hand side
                self.LinearModel.Ucon[self.nlin+self.np_ind+self.nq_ind:self.nlin+self.np_ind+self.nq_ind+self.nr_ind] = \
                        -self.cons_xk[self.r_ind] + np.dot(J_x_k,x_k) + model.Ucon[self.r_ind]
                        
                self.LinearModel.Ucon[self.nlin+self.np_ind+self.nq_ind+self.nr_ind:self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind] = \
                        self.cons_xk[self.r_ind] - np.dot(J_x_k,x_k) - model.Lcon[self.r_ind]


            if self.nz_ind > 0:
                J_x_k = self.J_x_k[self.z_ind]

                # Update constraint Matrix                 
                self.LinearModel.A[self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind:self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind+2*self.nz_ind,
                               :model.n] = np.concatenate((J_x_k,J_x_k))
                               
                # Update nonlinear linearized right hand side
                rhs = np.dot(J_x_k,x_k) - self.cons_xk[self.z_ind] + model.Lcon[self.z_ind]
    
                self.LinearModel.Lcon[self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind:self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind+self.nz_ind] = rhs
                self.LinearModel.Ucon[self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind+self.nz_ind:self.nlin+self.np_ind+self.nq_ind+2*self.nr_ind+2*self.nz_ind] = rhs

            # Update sparse matrix
            self.LP_solver.coefMatrix = csc_matrix(self.LinearModel.A)
            self.LP_solver.setRowLowerArray(self.LinearModel.Lcon)
            self.LP_solver.setRowUpperArray(self.LinearModel.Ucon)

        else:
            # Only linear constraint
            self.LinearModel.c = self.gFxk
            
        # Update bound constraints (commun for all configuration)
        self.LinearModel.Lvar[:model.n] = np.maximum(model.Lvar, x_k - self.trustRadius)
        self.LinearModel.Uvar[:model.n] = np.minimum(model.Uvar, x_k + self.trustRadius)

        self.LP_solver.setColumnLowerArray(self.LinearModel.Lvar)
        self.LP_solver.setColumnUpperArray(self.LinearModel.Uvar)
        self.LP_solver.setObjectiveArray(self.LinearModel.c)
        return


    def findStep(self):
        """
        Solve subproblem linearized and return the potential solution x_trial
        with CyLP
        """
        
        np.set_printoptions(threshold=np.nan)
        returnMessage = self.LP_solver.dual(startFinishOptions='x')
        if returnMessage not in ['optimal']:
            return self.x_k

        if self.model.nnln > 0:
            self.elastic_sol = self.LP_solver.primalVariableSolution[self.model.n:]

        return self.LP_solver.primalVariableSolution[:self.model.n]

    def projection(self):
        """ 
        Solve projection to find feasible solution in linear constraints
        """
        
        if self.nlin > 0:

            Lcon = np.hstack((-np.inf*np.ones(2*self.model.n), self.model.Lcon[self.model.lin]))
            Ucon = np.hstack((self.x_k, -self.x_k, self.model.Ucon[self.model.lin]))

            self.proj.setRowLowerArray(Lcon)
            self.proj.setRowUpperArray(Ucon) 
        else:
            self.proj.setRowUpperArray(np.hstack((self.x_k, -self.x_k)))

        returnMessage = self.proj.dual(startFinishOptions='x')
        self.x_k = self.proj.primalVariableSolution[self.model.n:]
        
        return


    def evaluateFunction(self, x):
        """
        Compute the penality function at x
        """

        if x is self.x_k:
            ret = self.Fxk
        else:
            ret = self.Fx_trial
        if self.model.nnln > 0:
            ret += self.mu * self.getInfeasibility(x)
        return ret

    def originalFunctionChange(self):
        """
        Compute the difference of penality function between x_k and x_trial
        """

        ret = self.Fxk - self.Fx_trial
        if self.model.nnln > 0:
            self.infeasibilityChange = self.inf_x_k - self.inf_x_trial
            ret += self.mu * self.infeasibilityChange
        return ret

    def getInfeasibility(self, x):
        """
        Return the sum of infeasibilities in the nonlinear constraints
        """
        if x is self.x_k:
            c = self.cons_xk
        else:
            c = self.cons_x_trial

        zeros = np.zeros(self.model.m)

        return np.sum(np.maximum(zeros, c - self.model.Ucon) + np.maximum(zeros, self.model.Lcon - c))


    def getLinearInfeasibility(self, x):
        """ 
        Return the sum of infeasibilities in the linear constraints and bounds
        """
        # Compute infeasibilities from the bounds
        zeros = np.zeros(self.model.n)
        inf_bounds = np.sum(np.maximum(zeros, x - self.model.Uvar) + np.maximum(zeros, self.model.Lvar - x))

        # Compute infeasibilities from the linear constraints
        if self.model.nlin >0:
            zeros = np.zeros(self.model.nlin)
            if x is self.x_k:
                c = self.cons_xk
            else:
                c = self.cons_x_trial

            inf_linear = np.sum(np.maximum(zeros, c[self.model.lin] - self.model.Ucon[self.model.lin]) + \
                         np.maximum(zeros, self.model.Lcon[self.model.lin] - c[self.model.lin]))

        else:
            inf_linear = 0.
        return inf_bounds+inf_linear

    def modelChange(self):
        """
        Compute the difference of linearized function between x_k and x_trial
        """
        ret = -np.dot(self.gFxk, self.step)
        if self.model.nnln > 0:
            ret += self.mu * self.inf_x_k
            ret -= self.mu * (np.sum(self.elastic_sol))
        return ret

    def computeOptimalitygap(self):
        """
        Compute Karush-Kuhn-Tucker (KKT) conditions representing
        the first order optimality conditions for nonlinear problem constrained .

        Return the maximum infinity norm of the KKT
        """

        # Initialize constraints and variables dual
        self.stored_cons_dual_xtrial = np.zeros(self.model.m)
        self.stored_var_dual_xtrial = np.zeros(self.model.n)

        # Get the constraints value at x_trial
        cons = self.cons_x_trial

        # Get active constraints in NLP
        Ucons_actives = np.isclose(cons, self.model.Ucon)
        Lcons_actives = np.isclose(cons, self.model.Lcon)

        # Get active boundaries in NLP
        Uvar_actives = np.isclose(self.x_trial, self.model.Uvar)
        Lvar_actives = np.isclose(self.x_trial, self.model.Lvar)

        if np.all(np.concatenate((Ucons_actives,Lcons_actives,Uvar_actives,Lvar_actives))== False):
            return norm(self.gFx_trial,np.inf)

        else:
            # Indice of constraints and bound actives
            cons_actives = np.arange(self.model.m)[Ucons_actives+Lcons_actives]
            var_indice = np.arange(self.model.n)

            # Create the matrix to compute the multipliers
            A = np.concatenate((self.J_x_trial[cons_actives,:],
                                np.eye(self.model.n)[Uvar_actives,:],
                                -np.eye(self.model.n)[Lvar_actives,:]))

            # solve the equation ax = b to estimate the multipliers
            self.pi = np.linalg.lstsq(A.T ,-self.gFx_trial)[0]

            Uvar_actives = var_indice[Uvar_actives]
            Lvar_actives = var_indice[Lvar_actives]

            self.stored_cons_dual_xtrial[cons_actives] = self.pi[:cons_actives.size]
            self.stored_var_dual_xtrial[Uvar_actives] = self.pi[cons_actives.size:cons_actives.size+Uvar_actives.size]
            self.stored_var_dual_xtrial[Lvar_actives] = self.pi[cons_actives.size+Uvar_actives.size:]

            return norm(np.concatenate((np.dot(self.pi,A)+self.gFx_trial, self.stored_cons_dual_xtrial*cons)),np.inf)

    def solve(self, model=None):
        """
        Solve method.
        """

        self.resetStoredValues()
        if model is not None:
            self.model = model
        self.x_k = self.model.x0

        # Check if the solution is feasible in linear constraints
        if self.getLinearInfeasibility(self.x_k) >= self.feas_tol:
            self.projection()


        if self.debug:
            self.log.debug("initial point =")
            self.log.debug(self.x_k)
            self.log.debug("initial obj = %f",self.Fxk )

            # Check derivative
            dcheck = DerivativeChecker(self.model, self.x_k)
            dcheck.check(hess = False, chess = False)

            self.log.debug("error(grad) =")
            self.log.debug(dcheck.grad_errs)
            self.log.debug("error(jac)  =")
            self.log.debug(dcheck.jac_errs)

        returnMessage = 'maximum iterations'

        while self.iteration < self.major_iter:

            if self.verbose and self.iteration % 10 == 0:
                len_info = 127
                header = ['It', 'Obj   ', 'KKT  ', '|x|','|y|','|z|',
                          'Ared', 'Pred', 'Ratio', 'Step', 'Radius', 'nlpInf', 'lpInf', 'mu']
                print len_info * '-'
                print self.header_fmt % tuple(header)
                print len_info * '-'

            self.iteration += 1

            self.updateSubproblem()

            self.x_trial = self.findStep()

            if self.debug:
                self.log.debug(70 * '-')
                self.log.debug('Iteration %d:',self.iteration)
                self.log.debug("x_k:")
                self.log.debug(self.x_k)
                self.log.debug("f(x_k):%f",self.Fxk )
                self.log.debug("inf_xk:%f",self.inf_x_k )
                self.log.debug("x_trial:")
                self.log.debug(self.x_trial)
                self.log.debug("f(x_trial):%f",self.Fx_trial )
                self.log.debug("inf_x_trial:%f",self.inf_x_trial )

            stepNorm = norm(self.step, ord=np.inf)

            # Compute ratio 
            pred = self.modelChange()
            ared = self.originalFunctionChange()
            ratio = ared / pred

            # Perform backtracking linesearch, if not succesful, reject the step 
            if not (ratio > self.eta1 and self.inf_x_trial <= self.feas_accept) and self.backtracking:
                slope = np.dot(self.step, self.gFxk)
                bk = 0
                alpha = 1.
                qk = self.l1_Fxk
                qk1 = self.l1_Fx_trial
                original_step = self.step
                while bk < self.backtrackingSegments and qk1 >= qk+ 1.0e-4 * alpha * slope:
                    bk += 1
                    alpha /= self.backtrackingMultiplier
                    self.x_trial = self.x_k + alpha * original_step
                    qk1 = self.l1_Fx_trial

                stepNorm *= alpha

            if ratio > self.eta1 and self.inf_x_trial <= self.feas_accept:
                # Check benefice
                if self.l1_Fxk - self.l1_Fx_trial > self.relChange_tol:
                    self.countNotProgressing = 0
                 
                # Step accpeted
                self.x_k = self.x_trial

                if self.debug:
                    self.log.debug("step accepted")

                if ratio > self.eta2:
                    # Increase trust region
                    self.trustRadius *= max(self.trustRadiusIncrease, stepNorm)

                    if self.debug:
                        self.log.debug("Trust-radius increased")

            else:
                # Step rejected, decrease trust region radius
                self.trustRadius = stepNorm*self.trustRadiusDecrease

                if self.backtracking and self.last_unsuccessful_bt_iter == self.iteration - 1:
                    self.consecutive_unsuccessful_bt_count += 1
                    if self.consecutive_unsuccessful_bt_count == 5:
                        self.backtrackingSegments = max(self.backtrackingSegments + 5 , 20)
                        self.consecutive_unsuccessful_bt_count = 0
                self.last_unsuccessful_bt_iter = self.iteration

                if self.debug:
                    self.log.debug("backtrack UNsuccessful")
                    self.log.debug("step rejected, trust radius decreased")

                # Count number of non-progress
                if self.l1_Fxk - self.l1_Fx_trial <= self.relChange_tol:
                    self.countNotProgressing += 1
                    if self.countNotProgressing >= self.nIterNoProgress:
                        returnMessage = 'Stopped on lack of progress (too many iteration without progress).'
                        break


            if stepNorm < self.step_tol and self.inf_x_trial <= self.feas_tol:
                # Optimality condition KKT
                # See Bazaraa, Sherali, Shetty
                returnMessage = 'Optimal'
                break

            self.updateMu()

            if self.verbose:

                # Print information every iteration
                info_line = [self.iteration, self.Fxk, self.optimalityGap_xk,
                             norm(self.x_k),norm(self.cons_dual),norm(self.var_dual),
                             ared, pred, ratio, stepNorm, self.trustRadius, self.inf_x_k,
                             np.sum(self.elastic_sol), self.mu]

                print self.fmt % tuple(info_line)

        return returnMessage

    def updateMu(self):
        '''
        Update the penalty parameter, mu with dual constraint from LP Model
         Do nothing if no nonlinear constraints present
        '''
        if self.model.nnln == 0:
            self.mu = 0.
            return
        if np.isclose(np.sum(self.elastic_sol),0):
            return
        dual_infinity_norm = norm(self.LP_solver.dualConstraintSolution, np.inf)
        self.mu = min(3 * dual_infinity_norm, self.maxMu)

        return

    def resetStoredValues(self):
        self.x_trial = None
        self.x_k = None
        self.stored_step = None
        self.stored_optimalityGap = None
        self.stored_Fxk = None
        self.stored_l1_Fxk = None
        self.stored_gFxk = None
        self.stored_Fx_trial = None
        self.stored_l1_Fx_trial = None
        self.stored_cons_xk = None
        self.stored_inf_x_k = None
        self.stored_inf_x_trial = None
        self.stored_cons_dual = np.inf
        self.stored_var_dual = np.inf
        self.stored_cons_dual_xtrial = None
        self.stored_var_dual_xtrial = None
        self.stored_J_x_k = None
        self.stored_J_x_trial = None
        self.stored_J_L = None
        self.iteration = 0
        self.elastic_sol = np.zeros(1)
        self.countNotProgressing = 0
        self.last_unsuccessful_bt_iter = -1
        self.consecutive_unsuccessful_bt_count = 0
        self.trustRadius = self.stored_trustradius_init
        self.mu = self.stored_mu_init

    @property
    def x_k(self):
        return self._x_k

    @x_k.setter
    def x_k(self, value):
        self._x_k = value
        self.stored_step = None
        self.stored_J_L = None
        if value is self._x_trial and self._x_trial is not None:
            self.stored_Fxk = self.stored_Fx_trial.copy()
            self.stored_l1_Fxk = self.stored_l1_Fx_trial.copy()
            self.stored_inf_x_k = copy.copy(self.stored_inf_x_trial)
            self.stored_cons_xk = np.copy(self.stored_cons_x_trial)
            self.stored_J_x_k = np.copy(self.J_x_trial)
            self.stored_gFxk = np.copy(self.gFx_trial)

            if self.verbose:
                # Compute dual informations
                self.optimalityGap
                self.stored_var_dual = np.copy(self.stored_var_dual_xtrial)
                self.stored_cons_dual = np.copy(self.stored_cons_dual_xtrial)
                self.stored_optimalityGap_xk = self.stored_optimalityGap.copy()

        else:
            self.stored_cons_xk = None
            self.stored_Fxk = None
            self.stored_l1_Fxk = None
            self.stored_gFxk = None
            self.stored_J_x_k = None
            self.stored_inf_x_k = None
            self.stored_optimalityGap_xk = np.inf
            self.stored_cons_dual = np.inf
            self.stored_var_dual = np.inf

    @property
    def x_trial(self):
        return self._x_trial

    @x_trial.setter
    def x_trial(self, value):
        if value is not None:
            self._x_trial = value.copy()
        else:
            self._x_trial = None
        self.stored_step = None
        self.stored_Fx_trial = None
        self.stored_l1_Fx_trial = None
        self.stored_cons_x_trial = None
        self.stored_gFx_trial = None
        self.stored_J_x_trial = None
        self.stored_optimalityGap = None
        self.stored_inf_x_trial = None
        self.stored_cons_dual_xtrial = None
        self.stored_var_dual_xtrial = None
        if self.model.nnln == 0:
            self.stored_inf_x_trial = np.zeros(1)


    @property
    def step(self):
        if self.stored_step is None:
            self.stored_step = self.x_trial - self.x_k
        return self.stored_step


    @property
    def optimalityGap_xk(self):
        return self.stored_optimalityGap_xk

    @property
    def optimalityGap(self):
        if self.stored_optimalityGap is None:
            self.stored_optimalityGap = self.computeOptimalitygap()
        return self.stored_optimalityGap

    @property
    def Fxk(self):
        if self.stored_Fxk is None:
            self.stored_Fxk = self.model.obj(self.x_k)
        return self.stored_Fxk

    @property
    def l1_Fxk(self):
        if self.stored_l1_Fxk is None:
            self.stored_l1_Fxk = self.evaluateFunction(self.x_k)
        return self.stored_l1_Fxk

    @property
    def gFxk(self):
        
        if self.stored_gFxk is None:
            self.stored_gFxk = self.model.grad(self.x_k)
        return self.stored_gFxk

    @property
    def cons_xk(self):
        if self.stored_cons_xk is None:
            self.stored_cons_xk = self.model.cons(self.x_k)
        return self.stored_cons_xk

    @property
    def cons_x_trial(self):
        if self.stored_cons_x_trial is None:
            self.stored_cons_x_trial = self.model.cons(self.x_trial)
        return self.stored_cons_x_trial

    @property
    def Fx_trial(self):
        if self.stored_Fx_trial is None:
            self.stored_Fx_trial = self.model.obj(self.x_trial)
        return self.stored_Fx_trial

    @property
    def l1_Fx_trial(self):
        if self.stored_l1_Fx_trial is None:
            self.stored_l1_Fx_trial = self.evaluateFunction(self.x_trial)
        return self.stored_l1_Fx_trial

    @property
    def gFx_trial(self):
        if self.stored_gFx_trial is None:
            self.stored_gFx_trial = self.model.grad(self.x_trial)
        return self.stored_gFx_trial

    @property
    def inf_x_k(self):
        if self.stored_inf_x_k is None:
            self.stored_inf_x_k = self.getInfeasibility(self.x_k)
        return self.stored_inf_x_k

    @property
    def inf_x_trial(self):
        if self.stored_inf_x_trial is None:
            self.stored_inf_x_trial = self.getInfeasibility(self.x_trial)
        return self.stored_inf_x_trial

    @property
    def cons_dual(self):
        return self.stored_cons_dual

    @property
    def var_dual(self):
        return self.stored_var_dual

    @property
    def cons_dual_xtrial(self):
        return self.stored_cons_dual_xtrial

    @property
    def var_dual_xtrial(self):
        return self.stored_var_dual_xtrial

    @property
    def J_x_k(self):
        if self.stored_J_x_k is None:
            self.stored_J_x_k = self.model.jac(self.x_k)
        return self.stored_J_x_k

    @property
    def J_x_trial(self):
        if self.stored_J_x_trial is None:
            self.stored_J_x_trial = self.model.jac(self.x_trial)
        return self.stored_J_x_trial

    @property
    def J_L(self):
        if self.stored_J_L is None:
            lci = self.model.lin
            if self.stored_J_x_k is None:
                self.stored_J_x_k = self.model.jac(self.x_k)
            self.stored_J_L = self.stored_J_x_k[lci, :]
        return self.stored_J_L

    @property
    def backtrackingSegments(self):
        return self._backtrackingSegments

    @backtrackingSegments.setter
    def backtrackingSegments(self, value):
        self._backtrackingSegments = value
        self.backtrackingMultiplier = 1 + 1. / value
