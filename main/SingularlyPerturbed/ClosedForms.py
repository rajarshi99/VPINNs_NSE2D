# %%
"""
@Title:
    hp-VPINNs: A General Framework For Solving PDEs
    Application to 2D Poisson Eqn

@author: 
    Ehsan Kharazmi
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu
"""

###############################################################################
###############################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time
import sys
import mlflow
import yaml
import os
import sys

np.random.seed(1234)
tf.random.set_seed(1234)

tf.compat.v1.disable_eager_execution()

###############################################################################
###############################################################################
class VPINN:
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad, U_exact_total, F_exact_total,\
                 gridx, gridy, N_testfcn, X_test, u_test, layers,input_data):

        self.x = X_u_train[:,0:1]
        self.y = X_u_train[:,1:2]
        self.utrain = u_train
        self.xquad  = X_quad[:,0:1]
        self.yquad  = X_quad[:,1:2]
        self.wquad  = W_quad
        self.xf = X_f_train[:,0:1]
        self.yf = X_f_train[:,1:2]
        self.ftrain = f_train
        self.xtest = X_test[:,0:1]
        self.ytest = X_test[:,1:2]
        self.utest = u_test
        self.Nelementx = np.size(N_testfcn[0])
        self.Nelementy = np.size(N_testfcn[1])
        self.Ntestx = N_testfcn[0][0]
        self.Ntesty = N_testfcn[1][0]
        self.U_ext_total = U_exact_total
        self.F_ext_total = F_exact_total
        self.input_data = input_data
        self.model_save_path = input_data["model_save_params"]['save_directory']

        # Track total time
        self.total_train_time = 0
        
        # Make directory for saving models
        if not os.path.exists(self.model_save_path + '/models'):
            os.makedirs(self.model_save_path + '/models')

        self.model_save_path = self.model_save_path + '/models'
       
        self.layers = layers
        self.weights, self.biases, self.a = self.initialize_NN(layers)
        
        self.x_tf     = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.y_tf     = tf.compat.v1.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.u_tf     = tf.compat.v1.placeholder(tf.float64, shape=[None, self.utrain.shape[1]])
        self.x_f_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.xf.shape[1]])
        self.y_f_tf = tf.compat.v1.placeholder(tf.float64, shape=[None, self.yf.shape[1]])
        self.f_tf   = tf.compat.v1.placeholder(tf.float64, shape=[None, self.ftrain.shape[1]])
        self.x_test = tf.compat.v1.placeholder(tf.float64, shape=[None, self.xtest.shape[1]])
        self.y_test = tf.compat.v1.placeholder(tf.float64, shape=[None, self.ytest.shape[1]])
        self.x_quad = tf.compat.v1.placeholder(tf.float64, shape=[None, self.xquad.shape[1]])
        self.y_quad = tf.compat.v1.placeholder(tf.float64, shape=[None, self.yquad.shape[1]])
                 
        self.u_pred_boundary = self.net_u(self.x_tf, self.y_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf)
        self.u_test = self.net_u(self.x_test, self.y_test)


        ## Obtain bx, by, c, and eps from input_data
        self.bx = input_data["bilinear_coefficients"]['bx']
        self.by = input_data["bilinear_coefficients"]['by']
        self.c  = input_data["bilinear_coefficients"]['c']
        self.eps = input_data["bilinear_coefficients"]['eps']
        self.boundary_loss_tau = input_data["bilinear_coefficients"]['boundary_loss_tau']
        self.stab_param_tau = input_data["bilinear_coefficients"]['stab_param_tau']
        

        B_x = tf.constant(self.bx, dtype=tf.float64)  ## Advection term in X direction
        B_y = tf.constant(self.by, dtype=tf.float64)  ## Advection term in Y direction
        C   = tf.constant(self.c, dtype=tf.float64)    ## reaction term
        STAB_PARAM_TAU = tf.constant(self.stab_param_tau, dtype=tf.float64)  ## Global regularization parameter
        EPSILON = tf.constant(self.eps, dtype=tf.float64)  ## Singular perturbation parameter
        
        self.varloss_total = 0
        for ex in range(self.Nelementx):
            for ey in range(self.Nelementy):
                F_ext_element  = self.F_ext_total[ex, ey]
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementy = N_testfcn[1][ey]
                
                x_quad_element = tf.constant(gridx[ex] + (gridx[ex+1]-gridx[ex])/2*(self.xquad+1))
                y_quad_element = tf.constant(gridy[ey] + (gridy[ey+1]-gridy[ey])/2*(self.yquad+1))
                jacobian_x     = ((gridx[ex+1]-gridx[ex])/2)
                jacobian_y     = ((gridy[ey+1]-gridy[ey])/2)
                jacobian       = ((gridx[ex+1]-gridx[ex])/2)*((gridy[ey+1]-gridy[ey])/2)
                
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)
                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element)
                d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element)
                                
                testx_quad_element = self.Test_fcnx(Ntest_elementx, self.xquad)
                d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                testy_quad_element = self.Test_fcny(Ntest_elementy, self.yquad)
                d1testy_quad_element, d2testy_quad_element = self.dTest_fcn(Ntest_elementy, self.yquad)
                
    
                
                if var_form == 0:
                    U_NN_element_Dif = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*integrand_1) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    U_NN_element_Adv = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*integrand_2) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    U_NN_element_SUPG = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*self.wquad[:,1:2]*((tau*U)*d1testx_quad_element[r]+(tau*V)*d1testy_quad_element[k])*integrand_3) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    U_NN_element = U_NN_element_Adv - U_NN_element_Dif - U_NN_element_SUPG

                if var_form == 1:
                    DUx_DVx        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    DUy_DVy = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    Diffusion_Term = (DUx_DVx + DUy_DVy) * EPSILON
                    
                    # Function for Convection term
                    DUx_V        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)    
                    
                    DUy_V        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    Convection_Term = B_x * DUx_V + B_y * DUy_V          
                    
                    ## Reaction Term
                    U_V  = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    Reaction_Term = U_V * C
                    
                    U_NN_element = Diffusion_Term + Convection_Term + Reaction_Term
                       

                    # U_NN_element_Diff = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                    #                 self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                    #                 for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    # U_NN_element_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                    #                 self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                    #                 for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    # U_NN_element = - U_NN_element_1 - U_NN_element_2

                ## Variational Form for SUPG
                if var_form == 2:
                    DUx_DVx        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    DUy_DVy = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    Diffusion_Term = (DUx_DVx + DUy_DVy) * EPSILON
                    
                    # Function for Convection term
                    DUx_V        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)    
                    
                    DUy_V        = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                     self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    Convection_Term = B_x * DUx_V + B_y * DUy_V
                    
                    ## Reaction Term
                    U_V  = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    Reaction_Term = U_V * C
                    
                    U_NN_element = Diffusion_Term + Convection_Term + Reaction_Term         
                    
                    
                    ## ----- SUPG Diffusion Term ----- ##
                    DDUx_V_BxDVx_ByDVy = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum( \
                                        self.wquad[:,0:1]*self.wquad[:,1:2]*d2xu_NN_quad_element*testx_quad_element[r] * testy_quad_element[k] *  \
                                        ( (B_x * d1testx_quad_element[r] * testy_quad_element[k])  + (B_y * d1testy_quad_element[k] * testx_quad_element[r]) ) )\
                                        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)

                    DDUy_V_BxDVx_ByDVy = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum( \
                                        self.wquad[:,0:1]*self.wquad[:,1:2]*d2yu_NN_quad_element*testx_quad_element[r] * testy_quad_element[k] *  \
                                        ( (B_x * d1testx_quad_element[r] * testy_quad_element[k])  + (B_y * d1testy_quad_element[k] * testx_quad_element[r]) ) )\
                                        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    SUPG_Diffusion_Term = (DDUx_V_BxDVx_ByDVy + DDUy_V_BxDVx_ByDVy) * -1.0 * EPSILON
                    
                    ## ----- SUPG Convection Term ----- ##
                    
                    SUPG_Convection_Term = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum( \
                                           self.wquad[:,0:1]*self.wquad[:,1:2] * \
                                           (B_x * d1xu_NN_quad_element + B_y * d1yu_NN_quad_element) * \
                                           ((B_x * d1testx_quad_element[r] * testy_quad_element[k])  + (B_y * d1testy_quad_element[k] * testx_quad_element[r]) ) ) 
                                             for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                                                               

                    ## --- Reaction Term --- ##
                    
                    SUPG_Reaction_Term = tf.convert_to_tensor([[jacobian*tf.reduce_sum( \
                                          self.wquad[:,0:1]*self.wquad[:,1:2] * \
                                          C * u_NN_quad_element * \
                                          ((B_x * d1testx_quad_element[r] * testy_quad_element[k])  + (B_y * d1testy_quad_element[k] * testx_quad_element[r]) ) ) 
                                          for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    
                    
                    
                    SUPG_Combined = STAB_PARAM_TAU*(SUPG_Diffusion_Term + SUPG_Convection_Term + SUPG_Reaction_Term)
                    
                    U_NN_element = U_NN_element + SUPG_Combined
                
                
                Res_NN_element = tf.reshape(U_NN_element - F_ext_element, [1,-1]) 
                # Res_NN_element = tf.reshape(U_NN_element, [1,-1]) 
                loss_element = tf.reduce_mean(tf.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element
 
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_pred_boundary))
        self.lossv = self.varloss_total
        self.lossp = tf.reduce_mean(tf.square(self.f_pred - self.ftrain))
        
        if scheme == 'VPINNs':
            self.loss  = self.boundary_loss_tau*self.lossb + self.lossv 
        if scheme == 'PINNs':
            self.loss  = self.boundary_loss_tau*self.lossb + self.lossp 
        

        self.learning_rate = self.input_data["model_run_params"]["learning_rate"]
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
#        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess = tf.compat.v1.Session()
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess.run(self.init)
        
###############################################################################
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.01, dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases, a
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim), dtype=np.float64)
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)
 
    
    def neural_net(self, X, weights, biases, a):
        num_layers = len(weights) + 1
        
        H = X 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y):  
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)
        return u

    def net_dxu(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu, d2xu

    def net_dyu(self, x, y):
        u   = self.net_u(x, y)
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        return d1yu, d2yu

    def net_f(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        # ftemp = d2xu + d2yu
        ftemp = 1
        return ftemp

    def Test_fcnx(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            test_total.append(test)
        return np.asarray(test_total)

    def Test_fcny(self, N_test,y):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
            test_total.append(test)
        return np.asarray(test_total)

    def dTest_fcn(self, N_test,x):
        d1test_total = []
        d2test_total = []
        for n in range(1,N_test+1):
            if n==1:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n==2:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
            else:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
        return np.asarray(d1test_total), np.asarray(d2test_total)

    def save_model(self, path,iteration):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess, path+ '/model_{}'.format(iteration))
        print("Model saved in file: %s" % save_path)
      

###############################################################################
    def train(self, nIter):
        
        tf_dict = {self.x_tf: self.x  , self.y_tf: self.y,
                   self.u_tf: self.utrain,
                   self.x_test: self.xtest, self.y_test: self.ytest,
                   self.x_f_tf: self.xf, self.y_f_tf: self.yf}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            loss_his.append(loss_value)
#            if it % 1 == 0:
#                loss_value = self.sess.run(self.loss, tf_dict)
#                u_pred     = self.sess.run(self.u_test, tf_dict)
#                u_pred_his.append(u_pred)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                self.total_train_time += elapsed
                str_print = ''.join(['It: %d, Loss: %.3e, Time: %.2f'])
                print(str_print % (it, loss_value, elapsed))
                start_time = time.time()
            if it % self.input_data["model_save_params"]["save_frequency"] == 0:
                self.save_model(self.model_save_path,it)
        
        self.save_model(self.model_save_path,it)

    def predict(self):
        u_pred = self.sess.run(self.u_test, {self.x_test: self.xtest, self.y_test: self.ytest})
        return u_pred


#%%
###############################################################################
# =============================================================================
#                               Main
# =============================================================================    
if __name__ == "__main__":     

    '''
    Hyper-parameters: 
        scheme     = is either 'PINNs' or 'VPINNs'
        Net_layer  = the structure of fully connected network
        var_form   = the form of the variational formulation used in VPINNs
                     0, 1, 2: no, once, twice integration-by-parts
        N_el_x, N_el_y     = number of elements in x and y direction
        N_test_x, N_test_y = number of test functions in x and y direction
        N_quad     = number of quadrature points in each direction in each element
        N_bound    = number of boundary points in the boundary loss
        N_residual = number of residual points in PINNs
    '''

    # Check if the number of arguments is correct
    if len(sys.argv) < 2:
        print('Usage: python3 hp-VPINN-Singularly_perturbed_2D.py <input yaml file> ' )
        exit(0)
    
    # Read the input yaml file
    with open(sys.argv[1], 'r') as f:
        input_data = yaml.safe_load(f)
   
    
    save_directory = input_data["model_save_params"]["save_directory"]

    ## Create a directory to save the output and the model files
    os.system(f"mkdir -p {save_directory}" )

    scheme = 'VPINNs'
    Net_layer = input_data["model_run_params"]["NN_model"]
    N_el_x = input_data["model_run_params"]["num_elements_x"]
    N_el_y = input_data["model_run_params"]["num_elements_y"]
    N_shape_func_x = input_data["model_run_params"]["num_shape_functions_x"]
    N_shape_func_y = input_data["model_run_params"]["num_shape_functions_y"]
    N_test_x = N_el_x*[N_shape_func_x]
    N_test_y = N_el_y*[N_shape_func_y]
    N_quad = input_data["model_run_params"]["num_quad_points"]
    N_bound = input_data["model_run_params"]["num_bound_points"]
    N_residual = input_data["model_run_params"]["num_residual_points"]   

    max_iter = input_data["model_run_params"]["max_iter"]
    

    # Assign BX and By
    b_x = input_data["bilinear_coefficients"]['bx']
    b_y = input_data["bilinear_coefficients"]['by']
    stab_param_tau = input_data["bilinear_coefficients"]['stab_param_tau']
    B_x = tf.convert_to_tensor(b_x, dtype=tf.float64)
    B_y = tf.convert_to_tensor(b_y, dtype=tf.float64) 

    # Assign Var form
    var_form = input_data["model_run_params"]["var_form"]
    if var_form == "VarPINNs":
        var_form = 1
    elif var_form == "VarPINNs_SUPG":
        var_form = 2
    else:
        print("Wrong var_form")
        exit(0)



    ###########################################################################
    def Test_fcn_x(n,x):
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
       return test
    def Test_fcn_y(n,y):
       test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
       return test

    def dTest_fcn( N_test,x):
        d1test_total = []
        d2test_total = []
        for n in range(1,N_test+1):
            if n == 1:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n==2:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
            else:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
        return np.asarray(d1test_total)
      

    ###########################################################################    
    omegax = 2*np.pi
    omegay = 2*np.pi
    r1 = 10
    
    # Read the csv file and save the exact Solution
    output_file = input_data["model_run_params"]["exact_solution_csv"]
    exact_solution = np.genfromtxt(output_file, delimiter=',')
    

    ## For u_ext and f_ext, the values are calculated by multiplying the incoming variables to preserve the shape.
    # Function which stores the exact solution , So it will be used to obtain boundary values also
    # For now, use the below for boundary and use the `u_exact` for computing loss
    def u_ext(x, y):
        # Check if the given arrays x and y are arrays or just single elements
        if np.isscalar(x) and np.isscalar(y):
            # If both x and y are single elements, assign the value of u_temp based on the value of y
            if np.isclose(y, 1):
                utemp = 1.0
            else:
                utemp = 0
        else:
            # If either x or y is an array, assign the value of u_temp based on the values of y
            utemp = np.zeros_like(y)
            utemp[np.isclose(y, 1)] = 1.0
        return utemp
    
    # The Exact Solution Needs to be loaded from an external File for computing the solution 
    # assume that the exact solution is stored in the `exact_solution` numpy array
    def u_exact(x, y):
        # return the last column of the exact_solution Array
        return exact_solution[:,-1].reshape(-1,1)
        
    
    
    def f_ext(x,y):
        # gtemp = (-0.1*(omegax**2)*np.sin(omegax*x) - (2*r1**2)*(np.tanh(r1*x))/((np.cosh(r1*x))**2))*np.sin(omegay*(y))\
                # +(0.1*np.sin(omegax*x) + np.tanh(r1*x)) * (-omegay**2 * np.sin(omegay*(y)) )
        gtemp = x*0.0 + 1.0
        return gtemp

    
    
    ###########################################################################
    # Boundary points
    
    x_up = lhs(1,N_bound)     ## Generate Values from 0 to 1 using Latin Hypercube Sampling
    y_up = np.empty(len(x_up))[:,None]
    y_up.fill(1)
    b_up = np.empty(len(x_up))[:,None]
    b_up = u_ext(x_up, y_up)
    x_up_train = np.hstack((x_up, y_up))
    u_up_train = b_up

    x_lo = lhs(1,N_bound)
    y_lo = np.empty(len(x_lo))[:,None]
    y_lo.fill(0)
    b_lo = np.empty(len(x_lo))[:,None]
    b_lo = u_ext(x_lo, y_lo)
    x_lo_train = np.hstack((x_lo, y_lo))
    u_lo_train = b_lo

    y_ri = lhs(1,N_bound)
    x_ri = np.empty(len(y_ri))[:,None]
    x_ri.fill(1)
    b_ri = np.empty(len(y_ri))[:,None]
    b_ri = u_ext(x_ri, y_ri)
    x_ri_train = np.hstack((x_ri, y_ri))
    u_ri_train = b_ri    

    y_le = lhs(1,N_bound)
    x_le = np.empty(len(y_le))[:,None]
    x_le.fill(0)
    b_le = np.empty(len(y_le))[:,None]
    b_le = u_ext(x_le, y_le)
    x_le_train = np.hstack((x_le, y_le))
    u_le_train = b_le    

    X_u_train = np.concatenate((x_up_train, x_lo_train, x_ri_train, x_le_train))
    u_train = np.concatenate((u_up_train, u_lo_train, u_ri_train, u_le_train))

    ###########################################################################
    # Residual points for PINNs
    grid_pt = lhs(2,N_residual)
    xf = grid_pt[:,0]
    yf = grid_pt[:,1]
    ff = np.asarray([ f_ext(xf[j],yf[j]) for j in range(len(yf))])
    X_f_train = np.hstack((xf[:,None],yf[:,None]))
    f_train = ff[:,None]

    ###########################################################################
    # Quadrature points
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    Y_quad, WY_quad   = (X_quad, WX_quad)
    xx, yy            = np.meshgrid(X_quad,  Y_quad)
    wxx, wyy          = np.meshgrid(WX_quad, WY_quad)
    XY_quad_train     = np.hstack((xx.flatten()[:,None],  yy.flatten()[:,None]))
    WXY_quad_train    = np.hstack((wxx.flatten()[:,None], wyy.flatten()[:,None]))

    ###########################################################################
    # Construction of RHS for VPINNs
    NE_x, NE_y = N_el_x, N_el_y
    # [x_l, x_r] = [-1, 1]
    # [y_l, y_u] = [-1, 1]
    [x_l, x_r] = [0, 1]
    [y_l, y_u] = [0, 1]
    delta_x    = (x_r - x_l)/NE_x
    delta_y    = (y_u - y_l)/NE_y
    grid_x     = np.asarray([ x_l + i*delta_x for i in range(NE_x+1)])
    grid_y     = np.asarray([ y_l + i*delta_y for i in range(NE_y+1)])

#    N_testfcn_total = [(len(grid_x)-1)*[N_test_x], (len(grid_y)-1)*[N_test_y]]
    N_testfcn_total = [N_test_x, N_test_y]
 
    #+++++++++++++++++++
    x_quad  = XY_quad_train[:,0:1]
    y_quad  = XY_quad_train[:,1:2]
    w_quad  = WXY_quad_train
    U_ext_total = []
    F_ext_total = []
    

    
    # %%
    
    for ex in range(NE_x):
        for ey in range(NE_y):
            Ntest_elementx  = N_testfcn_total[0][ex]
            Ntest_elementy  = N_testfcn_total[1][ey]
            
            x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1)
            y_quad_element = grid_y[ey] + (grid_y[ey+1]-grid_y[ey])/2*(y_quad+1)
            jacobian       = ((grid_x[ex+1]-grid_x[ex])/2)*((grid_y[ey+1]-grid_y[ey])/2)
            
            testx_quad_element = np.asarray([ Test_fcn_x(n,x_quad)  for n in range(1, Ntest_elementx+1)])

            d1testx_quad_element = dTest_fcn(Ntest_elementx, x_quad)

            testy_quad_element = np.asarray([ Test_fcn_y(n,y_quad)  for n in range(1, Ntest_elementy+1)])
            d1testy_quad_element = dTest_fcn(Ntest_elementy,y_quad)
            
    
            u_quad_element = u_ext(x_quad_element, y_quad_element)
            f_quad_element = f_ext(x_quad_element, y_quad_element)
            
            U_ext_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*u_quad_element) \
                            for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            if var_form == 1:
                ## Normal Weak form
                F_ext_element = np.asarray([[jacobian*np.sum(\
                                w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) \
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            
            if var_form == 2:
                ## SUPG form
                F_ext_element_normal = np.asarray([[jacobian*np.sum(\
                                w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) \
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])

                F_ext_element_SUPG = np.asarray([[jacobian*np.sum(\
                                w_quad[:,0:1]*w_quad[:,1:2]*f_quad_element *  \
                                ( (b_x* d1testx_quad_element[r] * testy_quad_element[k])  + (b_y * d1testy_quad_element[k] * testx_quad_element[r]) ) ) 
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            
                F_ext_element = F_ext_element_normal + stab_param_tau * F_ext_element_SUPG
            U_ext_total.append(U_ext_element)
    
            F_ext_total.append(F_ext_element)
    
#    U_ext_total = np.reshape(U_ext_total, [NE_x, NE_y, N_test_y, N_test_x])
    F_ext_total = np.reshape(F_ext_total, [NE_x, NE_y, N_test_y[0], N_test_x[0]])
    

    ###########################################################################
    # Test points
    delta_test = 0.01
    xtest = np.arange(x_l, x_r + delta_test, delta_test)
    ytest = np.arange(y_l, y_u + delta_test, delta_test)
    
    # Thivin - Data_temp is modified to not to compute exact value using the given function
    data_temp = np.asarray([[ [xtest[i],ytest[j],0.0] for i in range(len(xtest))] for j in range(len(ytest))])
    
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    #  Exact = data_temp.flatten()[2::3]   # Thivin -- removed to read from external file
    Exact = u_exact(Xtest, Xtest)
    
    X_test = np.hstack((Xtest[:,None],Ytest[:,None]))
    u_test = Exact[:,None]
    
    


    ###########################################################################
    model = VPINN(X_u_train, u_train, X_f_train, f_train, XY_quad_train, WXY_quad_train,\
                  U_ext_total, F_ext_total, grid_x, grid_y, N_testfcn_total, X_test, u_test, Net_layer,input_data)
    

    u_pred_his, loss_his = [], []
    model.train(max_iter + 1)
    u_pred = model.predict()

#%%
    ###########################################################################
    # =============================================================================
    #    Plotting
    # =============================================================================
    from matplotlib import rc
    from cycler import cycler
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.prop_cycle'] = cycler(color=['darkblue', '#d62728', '#2ca02c', '#ff7f0e', '#bcbd22', '#8c564b', '#17becf', '#9467bd', '#e377c2', '#7f7f7f'])

    fig = plt.figure(1, figsize=(6.4,4.8))
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel(r'$iteration$')
    plt.ylabel(r'$loss \,\, values$')
    plt.yscale(r'log')
    plt.grid(True)
    plt.plot(loss_his)
    plt.tick_params()
    #fig.tight_layout()
    plt.title(r'Residual loss for $\epsilon$ = %s' % (input_data["bilinear_coefficients"]['eps']))
    plt.savefig(f"{save_directory}/loss_history.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/loss_history.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/loss_history.pdf", dpi=300, bbox_inches='tight')
    
    ###########################################################################
    x_train_plot, y_train_plot = zip(*X_u_train)
    x_f_plot, y_f_plot = zip(*X_f_train)
    fig, ax = plt.subplots(1)
    if scheme == 'VPINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        for xc in grid_x:
            plt.axvline(x=xc, ymin=0.045, ymax=0.954, linewidth=1.5)
        for yc in grid_y:
            plt.axhline(y=yc, xmin=0.045, xmax=0.954, linewidth=1.5)
    if scheme == 'PINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        plt.scatter(x_f_plot,y_f_plot)
        plt.axhline(-1, linewidth=1, linestyle='--', color='k')
        plt.axhline(1, linewidth=1, linestyle='--', color='k')
        plt.axvline(-1, linewidth=1, linestyle='--', color='k')
        plt.axvline(1, linewidth=1, linestyle='--', color='k')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    #ax.set_aspect(1)
    ax.locator_params(nbins=5)
    plt.tick_params()
    #fig.tight_layout()
    fig.set_size_inches(w=6.4, h=4.8)
    plt.title(r'Boundary Points for $\epsilon$ = %s' % (input_data["bilinear_coefficients"]['eps']))
    plt.savefig(f"{save_directory}/training_points.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/training_points.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/training_points.pdf", dpi=300, bbox_inches='tight')
    
    ###########################################################################
    x_test_plot = np.asarray(np.split(X_test[:,0:1].flatten(),len(ytest)))
    y_test_plot = np.asarray(np.split(X_test[:,1:2].flatten(),len(ytest)))
    u_test_plot = np.asarray(np.split(u_test.flatten(),len(ytest)))
    u_pred_plot = np.asarray(np.split(u_pred.flatten(),len(ytest)))
    
   

    fig_ext, ax_ext = plt.subplots(constrained_layout=True, figsize=(6.4, 4.8))
    CS_ext = ax_ext.contourf(x_test_plot, y_test_plot, u_test_plot, 100, cmap='jet', origin='lower')
    cbar = fig_ext.colorbar(CS_ext, shrink=0.85)
    cbar.ax.tick_params()
    ax_ext.locator_params(nbins=8)
    ax_ext.set_xlabel('$x$')
    ax_ext.set_ylabel('$y$')
    plt.tick_params( )
    ax_ext.set_aspect(1)
    #fig.tight_layout()
    plt.title(r'Exact Solution for $\epsilon$ = %s' % (input_data["bilinear_coefficients"]['eps']))
    plt.savefig(f"{save_directory}/Exact_Solution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Exact_Solution.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Exact_Solution.pdf", dpi=300, bbox_inches='tight')
    
    
    
    fig_pred, ax_pred = plt.subplots(constrained_layout=True, figsize=(6.4, 4.8))
    CS_pred = ax_pred.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred.colorbar(CS_pred, shrink=0.85)
    cbar.ax.tick_params()
    ax_pred.locator_params(nbins=8)
    ax_pred.set_xlabel('$x$')
    ax_pred.set_ylabel('$y$')
    plt.tick_params( )
    ax_pred.set_aspect(1)
    #fig.tight_layout()
    fig_pred.set_size_inches(w=6.4, h=4.8)
    plt.title(r'Predicted Solution for $\epsilon$ = %s' % (input_data["bilinear_coefficients"]['eps']))
    plt.savefig(f"{save_directory}/Predicted_Solution.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Predicted_Solution.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Predicted_Solution.pdf", dpi=300, bbox_inches='tight')
    
    
    fig_err, ax_err = plt.subplots(constrained_layout=True, figsize=(6.4, 4.8))
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), 100, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.85, format="%.4f")
    cbar.ax.tick_params()
    ax_err.locator_params(nbins=8)
    ax_err.set_xlabel('$x$')
    ax_err.set_ylabel('$y$')
    plt.tick_params( )
    ax_err.set_aspect(1)
    #fig.tight_layout()
    fig_err.set_size_inches(w=6.4, h=4.8)
    plt.title(r'Absolute Error for $\epsilon$ = %s' % (input_data["bilinear_coefficients"]['eps']))
    plt.savefig(f"{save_directory}/Absolute_Error.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Absolute_Error.eps", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_directory}/Absolute_Error.pdf", dpi=300, bbox_inches='tight')


    # Save the x_test_plot, y_test_plot, u_test_plot, u_pred_plot
    np.savez(f"{save_directory}/data.npz", x_test_plot=x_test_plot, y_test_plot=y_test_plot, u_test_plot=u_test_plot, u_pred_plot=u_pred_plot)
    

    # Compute the Error Between predicted and the actual 
    l2_error = np.linalg.norm(u_test_plot - u_pred_plot, 2)/np.linalg.norm(u_test_plot, 2)
    l1_error = np.linalg.norm(u_test_plot - u_pred_plot, 1)/np.linalg.norm(u_test_plot, 1)
    linf_error = np.linalg.norm(u_test_plot - u_pred_plot, np.inf)/np.linalg.norm(u_test_plot, np.inf)

    # Print all the Errors 
    print("-" * 82)
    print(f"| {'Summary of Training':^78} |")
    print("-" * 82)
    print(f"| {'Parameter':<20} | {'Value':<55} |")
    print("|" + "-" * 80 + "|")
    print(f"| {'L2 Error':<20} | {l2_error:<55} |")
    print(f"| {'L1 Error':<20} | {l1_error:<55} |")
    print(f"| {'Linf Error':<20} | {linf_error:<55} |")
    print(f"| {'Train Loss':<20} | {loss_his[-1]:<55} |")
    print(f"| {'Train Time':<20} | {model.total_train_time:<55} |")
    print("-" * 82)
    
    
    
    
    

    # if use mlflow
    if input_data['mlflow_parameters']['use_mlflow']:
        print("MLflow Version:", mlflow.version.VERSION)
        print("Tracking URI:", mlflow.tracking.get_tracking_uri())
        mlflow.set_experiment(f"{input_data['mlflow_parameters']['mlflow_experiment_name']}")
           
        # Setname
        mlflow_run_prefix = input_data["mlflow_parameters"]["mlflow_run_prefix"]
        
        # Get the current date and time as a string
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # Set mlflow run name
        mlflow.set_tag("mlflow.runName", mlflow_run_prefix + "_" + str(now))
        
        # create a plot directory
        os.system(f"mkdir -p {save_directory}/plots")
        os.system(f"mv {save_directory}/*png {save_directory}/*pdf {save_directory}/*eps  {save_directory}/plots")

        # Save the final numpy arrays
        os.system(f"mkdir -p {save_directory}/solution_arrays")
        os.system(f"mv {save_directory}/data.npz {save_directory}/solution_arrays")

        os.system(f"cp input.yaml {save_directory}/input.yaml")

        mlflow.log_metric("loss", loss_his[-1])
        mlflow.log_metric("train_time", model.total_train_time)
        mlflow.log_metric("time_per_iter", model.total_train_time/input_data["model_run_params"]["max_iter"])
        
        # log the Errors
        mlflow.log_metric("l2_error", l2_error)
        mlflow.log_metric("l1_error", l1_error)
        mlflow.log_metric("linf_error", linf_error)
        

        mlflow.log_param("date", now)

        # Log the parameters from the YAML file
        mlflow.log_param("max_iter", input_data["model_run_params"]["max_iter"])
        mlflow.log_param("NN_model", input_data["model_run_params"]["NN_model"])
        mlflow.log_param("var_form", input_data["model_run_params"]["var_form"])
        mlflow.log_param("boundary_loss_tau", input_data["model_run_params"]["boundary_loss_tau"])
        mlflow.log_param("num_quad_points", input_data["model_run_params"]["num_quad_points"])
        mlflow.log_param("num_bound_points", input_data["model_run_params"]["num_bound_points"])
        mlflow.log_param("num_residual_points", input_data["model_run_params"]["num_residual_points"])
        mlflow.log_param("num_elements_x", input_data["model_run_params"]["num_elements_x"])
        mlflow.log_param("num_elements_y", input_data["model_run_params"]["num_elements_y"])
        mlflow.log_param("num_shape_functions_x", input_data["model_run_params"]["num_shape_functions_x"])
        mlflow.log_param("num_shape_functions_y", input_data["model_run_params"]["num_shape_functions_y"])
        mlflow.log_param("learning_rate", input_data["model_run_params"]["learning_rate"])

        mlflow.log_param("eps", input_data["bilinear_coefficients"]["eps"])
        mlflow.log_param("bx", input_data["bilinear_coefficients"]["bx"])
        mlflow.log_param("by", input_data["bilinear_coefficients"]["by"])
        mlflow.log_param("c", input_data["bilinear_coefficients"]["c"])
        mlflow.log_param("stab_param_tau", input_data["bilinear_coefficients"]["stab_param_tau"])
        mlflow.log_param("boundary_loss_tau", input_data["bilinear_coefficients"]["boundary_loss_tau"])


        mlflow.log_artifact(f"{save_directory}/plots",artifact_path="plots")
        mlflow.log_artifact(f"{save_directory}/input.yaml")
        mlflow.log_artifact(f"{save_directory}/models",artifact_path="models")
        mlflow.log_artifact(f"{save_directory}/solution_arrays",artifact_path="solution_arrays")