import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.98 # discount factor

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.035 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax
        par.has_spouse = 1
        par.theta = 0.05

        # children
        par.p_birth = 0.1

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 30 # number of grid points in wealth grid    

        par.Nn = 2 # number of children

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        # spouse
        par.num_s = 2
        par.p_s = 0.8


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Na,par.Nk,par.num_s)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)

        sim.s = np.zeros(shape,dtype=np.int_)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

        sim.s_init = np.zeros(par.simN,dtype=np.int_)

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):
            for i_s, s in enumerate([0, 1]):

            # i. loop over state variables: number of children, human capital and wealth in beginning of period
             for i_n,kids in enumerate(par.n_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_n,i_a,i_k,i_s)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.

                        if t==par.T-1: # last period
                            
                            obj = lambda x: self.obj_last(x[0],assets,capital,kids,s,t)

                            # call optimizer
                            spouse_income = s * (0.1 + 0.01 * t)
                            hours_min = np.fmax( - (assets + spouse_income) / self.wage_func(capital,t) + 1.0e-5 , 0.0) # minimum amount of hours that ensures positive consumption
                            init_h = np.maximum(hours_min,2.0) if i_a==0 else np.array([sol.h[t,i_n,i_a-1,i_k,i_s]])
                            res = minimize(obj,init_h,bounds=((hours_min,np.inf),),method='L-BFGS-B')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital,kids,s,t)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                        else:
                            
                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[1],x[0],assets,capital,kids,s,t)  

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_h,ub_h),(lb_c,ub_c))
                
                            # call optimizer
                            idx_last = (t+1,i_n,i_a,i_k,i_s)
                            init = np.array([sol.h[idx_last],sol.c[idx_last]])
                            res = minimize(obj,init,bounds=bounds,method='L-BFGS-B',tol=1.0e-8) 
                        
                            # store results
                            sol.h[idx] = res.x[0]
                            sol.c[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,kids,s,t):
        par = self.par

        income = self.wage_func(capital, t) * hours
        y_spouse = s * (0.1 + 0.01 * t)
        cons = assets + income + (1.0 - par.tau) * y_spouse - par.theta * kids
        return cons

    def obj_last(self,hours,assets,capital,kids,s,t):
        cons = self.cons_last(hours,assets,capital,kids,s,t)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,kids,s,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours
        y_spouse = s * (0.1 + 0.01 * t)
        a_next = (1.0+par.r)*(assets + income + (1.0 - par.tau) * y_spouse - cons- par.theta * kids)
        k_next = capital + hours

        EV_next = 0.0
        for s_next in [0, 1]:
            prob_s = par.p_s if s_next == 1 else (1.0 - par.p_s)
            # no birth
            kids_next_no = kids
            V_next_no = sol.V[t+1, kids_next_no, :, :, s_next] 
            V_next_no_interp = interp_2d(par.a_grid, par.k_grid, V_next_no, a_next, k_next)

            # birth
            if (kids >= (par.Nn - 1)):
                V_next_birth_interp = V_next_no_interp
            else:
                kids_next_birth = kids + 1
                V_next_birth = sol.V[t+1, kids_next_birth, :, :, s_next] 
                V_next_birth_interp = interp_2d(par.a_grid, par.k_grid, V_next_birth, a_next, k_next)

            if s == 1:
                EV_given_s = par.p_birth * V_next_birth_interp + (1.0 - par.p_birth) * V_next_no_interp
            else:
                EV_given_s = V_next_no_interp

            EV_next += prob_s * EV_given_s
 


        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w_vec[t] * (1.0 + par.alpha * capital)

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            sim.s[i,0] = 1 if np.random.uniform() <= par.p_s else 0

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                c_matrix = sol.c[t, sim.n[i,t], :, :, sim.s[i,t]]
                h_matrix = sol.h[t, sim.n[i,t], :, :, sim.s[i,t]]
                sim.c[i,t] = interp_2d(par.a_grid, par.k_grid, c_matrix, sim.a[i,t], sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid, par.k_grid, h_matrix, sim.a[i,t], sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    y_spouse = sim.s[i,t] * (0.1 + 0.01 * t)
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income + (1.0 - par.tau) * y_spouse - sim.c[i,t] - par.theta * sim.n[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    birth = 0 
                    if (sim.draws_uniform[i,t] <= par.p_birth) and (sim.n[i,t] < (par.Nn-1)) and (sim.s[i,t] == 1):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth

                    sim.s[i,t+1] = 1 if np.random.uniform() <= par.p_s else 0
                    


