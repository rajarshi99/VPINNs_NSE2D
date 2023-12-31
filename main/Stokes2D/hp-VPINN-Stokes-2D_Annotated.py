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
# import tensorflow as tf
# %%
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time
from rich.console import Console
from rich.progress import track
from rich.table import Table

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


np.random.seed(1234)
tf.set_random_seed(1234)
tf.disable_eager_execution()
tf.disable_v2_behavior()

import inspect

def get_variable_info(variable):
    variable_type = type(variable).__name__

    if isinstance(variable, np.ndarray):
        variable_shape = variable.shape
    elif isinstance(variable, tf.Variable):
        variable_shape = variable.shape
    else:
        variable_shape = ""

    return variable_type, variable_shape



###############################################################################
###############################################################################
# %%
class VPINN:
    def __init__(self, coord_bound_train, u_bound_train, v_bound_train, coord_train_force, f_1_train, f_2_train, coord_quad_train, quad_weight_train, F_1_exact_total, F_2_exact_total,\
                 gridx, gridy, num_total_testfunc,coord_test, u_test,v_test,p_test, layers):

        self.x_bound = coord_bound_train[:,0:1] # boundary points in x direction. Data type: numpy.ndarray, Size: (4*N_bound,1)
        self.y_bound = coord_bound_train[:,1:2] # boundary points in y direction. Data type: numpy.ndarray, Size: (4*N_bound,1)
        self.u_bound = u_bound_train            # boundary values. Data type: numpy.ndarray, Size: (4*N_bound,1)
        self.v_bound = v_bound_train            # boundary values. Data type: numpy.ndarray, Size: (4*N_bound,1)

        self.x_quad  = coord_quad_train[:,0:1] # quadrature points in x direction. Data type: numpy.ndarray, Size: (N_quad*N_quad,1)
        self.y_quad  = coord_quad_train[:,1:2] # quadrature points in y direction. Data type: numpy.ndarray, Size: (N_quad*N_quad,1)
        self.w_quad  = quad_weight_train       # quadrature weights. Data type: numpy.ndarray, Size: (N_quad*N_quad,2)

        self.x_f = coord_train_force[:,0:1]    # force sampling points in x direction. Data type: numpy.ndarray, Size: (N_force,1)
        self.y_f = coord_train_force[:,1:2]    # force sampling points in y direction. Data type: numpy.ndarray, Size: (N_force,1)
        self.f1train = f_1_train                  # force values. Data type: numpy.ndarray, Size: (N_force,1)

        self.f2train = f_2_train                  # force values. Data type: numpy.ndarray, Size: (N_force,1)

        self.x_test = coord_test[:,0:1]         # test points in x direction. Data type: numpy.ndarray, Size: (N_test*N_test,1)
        self.y_test = coord_test[:,1:2]         # test points in y direction. Data type: numpy.ndarray, Size: (N_test*N_test,1)
        self.test_u = u_test                    # test values. Data type: numpy.ndarray, Size: (N_test*N_test,1)
        self.test_v = v_test                    # test values. Data type: numpy.ndarray, Size: (N_test*N_test,1)
        self.test_p = p_test                    # test values. Data type: numpy.ndarray, Size: (N_test*N_test,1)

        self.num_element_x = np.size(num_total_testfunc[0]) # number of elements in x direction. Data type: int, Size: (1)
        self.num_element_y = np.size(num_total_testfunc[1])  # number of elements in y direction. Data type: int, Size: (1)
        self.num_test_x = num_total_testfunc[0][0]          # number of test functions in x direction. Data type: int, Size: (1)
        self.num_test_y = num_total_testfunc[1][0]          # number of test functions in y direction. Data type: int, Size: (1)

        self.F_1_ext_total = F_1_exact_total                    # exact force values. Data type: numpy.ndarray, Size: (N_bound,1)
        self.F_2_ext_total = F_2_exact_total                    # exact force values. Data type: numpy.ndarray, Size: (N_bound,1)

       
        self.layers = layers                                # neural network structure. Data type: list, Size: (num_layers+1,1)
        self.weights, self.biases, self.a = self.initialize_NN(layers) # initialize the weights and biases of the neural network
        
        self.x_bound_tensor     = tf.placeholder(tf.float64, shape=[None, self.x_bound.shape[1]]) # boundary points in x direction. Data type: tf.placeholder, Size: (4*N_bound,1)
        self.y_bound_tensor     = tf.placeholder(tf.float64, shape=[None, self.y_bound.shape[1]]) # boundary points in y direction. Data type: tf.placeholder, Size: (4*N_bound,1)

        self.u_bound_tensor     = tf.placeholder(tf.float64, shape=[None, self.u_bound.shape[1]]) # boundary values. Data type: tf.placeholder, Size: (4*N_bound,1)
        self.v_bound_tensor     = tf.placeholder(tf.float64, shape=[None, self.v_bound.shape[1]]) # boundary values. Data type: tf.placeholder, Size: (4*N_bound,1)

        self.x_force_tensor = tf.placeholder(tf.float64, shape=[None, self.x_f.shape[1]]) # force sampling points in x direction. Data type: tf.placeholder, Size: (N_force,1)
        self.y_force_tensor = tf.placeholder(tf.float64, shape=[None, self.y_f.shape[1]]) # force sampling points in y direction. Data type: tf.placeholder, Size: (N_force,1)

        self.force_1_tensor   = tf.placeholder(tf.float64, shape=[None, self.f1train.shape[1]]) # force values. Data type: tf.placeholder, Size: (N_force,1)
        self.force_2_tensor   = tf.placeholder(tf.float64, shape=[None, self.f2train.shape[1]]) # force values. Data type: tf.placeholder, Size: (N_force,1)

        self.x_test_tensor = tf.placeholder(tf.float64, shape=[None, self.x_test.shape[1]]) # test points in x direction. Data type: tf.placeholder, Size: (N_test,1)
        self.y_test_tensor = tf.placeholder(tf.float64, shape=[None, self.y_test.shape[1]]) # test points in y direction. Data type: tf.placeholder, Size: (N_test,1)
        self.x_quad_tensor = tf.placeholder(tf.float64, shape=[None, self.x_quad.shape[1]]) # quadrature points in x direction. Data type: tf.placeholder, Size: (N_quad,1)
        self.y_quad_tensor = tf.placeholder(tf.float64, shape=[None, self.y_quad.shape[1]]) # quadrature points in y direction. Data type: tf.placeholder, Size: (N_quad,1)



        self.sess = tf.Session() # initialize the tensorflow session
        
        self.mew = 1.0

        #check
        self.u_pred_boundary = self.net_u(self.x_bound_tensor, self.y_bound_tensor)                 # predicted values at the boundary points. Data type: tf.tensor, Size: (N_bound,1)
        self.v_pred_boundary = self.net_v(self.x_bound_tensor, self.y_bound_tensor)                 # predicted values at the boundary points. Data type: tf.tensor, Size: (N_bound,1)
        self.f_pred = self.net_f(self.x_force_tensor, self.y_force_tensor)                          # predicted values at the force sampling points. Data type: tf.tensor, Size: (N_force,1)
        self.u_test = self.net_u(self.x_test, self.y_test)                                          # predicted values at the test points. Data type: tf.tensor, Size: (N_test,1)
        self.v_test = self.net_v(self.x_test, self.y_test)                                          # predicted values at the test points. Data type: tf.tensor, Size: (N_test,1)
        self.p_test = self.net_p(self.x_test, self.y_test)                                           # predicted values at the test points. Data type: tf.tensor, Size: (N_test,1)
    
        
        self.varloss_total = 0 # initialize the variational loss 
        
        for ex in range(self.num_element_x): # loop over the elements in x direction
            for ey in range(self.num_element_y): # loop over the elements in y direction

                F_1_ext_element  = self.F_1_ext_total[ex, ey]   # exact force values. Data type: numpy.ndarray, Size: (1,1)
                F_2_ext_element  = self.F_2_ext_total[ex, ey]   # exact force values. Data type: numpy.ndarray, Size: (1,1)

                Ntest_elementx = num_total_testfunc[0][ex]  # number of test functions in x direction. Data type: int, Size: (1)
                Ntest_elementy = num_total_testfunc[1][ey]  # number of test functions in y direction. Data type: int, Size: (1)

                x_quad_element = tf.constant(gridx[ex] + (gridx[ex+1]-gridx[ex])/2*(self.x_quad+1)) # quadrature points in x direction. Data type: tf.constant, Size: (N_quad,1)
                y_quad_element = tf.constant(gridy[ey] + (gridy[ey+1]-gridy[ey])/2*(self.y_quad+1)) # quadrature points in y direction. Data type: tf.constant, Size: (N_quad,1)

                jacobian_x     = ((gridx[ex+1]-gridx[ex])/2)                                        # jacobian in x direction. Data type: float, Size: (1)
                jacobian_y     = ((gridy[ey+1]-gridy[ey])/2)                                        # jacobian in y direction. Data type: float, Size: (1)
                jacobian       = ((gridx[ex+1]-gridx[ex])/2)*((gridy[ey+1]-gridy[ey])/2)            # jacobian. Data type: float, Size: (1)
                
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)                      # predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
                v_NN_quad_element = self.net_v(x_quad_element, y_quad_element)                      # predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
                p_NN_quad_element = self.net_p(x_quad_element, y_quad_element)                      # predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)


                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
                d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
                d1xv_NN_quad_element, d2xv_NN_quad_element = self.net_dxv(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
                d1yv_NN_quad_element, d2yv_NN_quad_element = self.net_dyv(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)


                testx_quad_element = self.Test_fcnx(Ntest_elementx, self.x_quad)                              # test functions in x direction. Data type: numpy.ndarray, Size: (Ntest_elementx,N_quad)
                d1testx_quad_element, d2testx_quad_element = self.grad_test_func(Ntest_elementx, self.x_quad) # first and second derivatives of the test functions in x direction. Data type: numpy.ndarray, Size: (Ntest_elementx,N_quad)
                testy_quad_element = self.Test_fcny(Ntest_elementy, self.y_quad)                              # test functions in y direction. Data type: numpy.ndarray, Size: (Ntest_elementy,N_quad)
                d1testy_quad_element, d2testy_quad_element = self.grad_test_func(Ntest_elementy, self.y_quad) # first and second derivatives of the test functions in y direction. Data type: numpy.ndarray, Size: (Ntest_elementy,N_quad)


                
    
                if var_form == 3:
                    # \int_{\Omega} \epsilon * \grad{c} \cdot \grad{v} dx dy = \int_{\Omega} \epsilon *  d/dx(c)* d/dx(vx) * vy dx dy
                    # \sum
                    U_NN_element_diff_1 = tf.convert_to_tensor([[self.mew*jacobian/jacobian_x*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*d1testx_quad_element[r]*self.w_quad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    

                    U_NN_element_diff_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*testx_quad_element[r]*self.w_quad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    V_NN_element_diff_1 = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*d1testx_quad_element[r]*self.w_quad[:,1:2]*testy_quad_element[k]*d1xv_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    V_NN_element_diff_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*testx_quad_element[r]*self.w_quad[:,1:2]*d1testy_quad_element[k]*d1yv_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    P_NN_element_1 = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*d1testx_quad_element[r]*self.w_quad[:,1:2]*testy_quad_element[k]*p_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    P_NN_element_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*testx_quad_element[r]*self.w_quad[:,1:2]*d1testy_quad_element[k]*p_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    Continuity_NN_element = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.w_quad[:,0:1]*testx_quad_element[r]*self.w_quad[:,1:2]*testy_quad_element[k]*(d1xu_NN_quad_element+d1yv_NN_quad_element)) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    P_NN_element_3 = tf.convert_to_tensor(tf.reduce_sum(\
                                    self.w_quad[:,0:1]*self.w_quad[:,1:2]*p_NN_quad_element) \
                                    , dtype= tf.float64) # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    
                    

                    U_NN_element = U_NN_element_diff_1 + U_NN_element_diff_2 # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)
                    V_NN_element = V_NN_element_diff_1 + V_NN_element_diff_2 # Data type: tf.tensor, Size: (Ntest_elementx,Ntest_elementy)

                Res_NN_element_1 = tf.reshape(U_NN_element - P_NN_element_1 - F_1_ext_element, [1,-1]) # residual. Data type: tf.tensor, Size: (1,Ntest_elementx*Ntest_elementy) 
                Res_NN_element_2 = tf.reshape(V_NN_element - P_NN_element_2 - F_2_ext_element, [1,-1]) # residual. Data type: tf.tensor, Size: (1,Ntest_elementx*Ntest_elementy)
                Res_NN_element_3 = tf.reshape(Continuity_NN_element, [1,-1]) # residual. Data type: tf.tensor, Size: (1,Ntest_elementx*Ntest_elementy)
                Res_NN_element_4 = tf.reshape(P_NN_element_3, [1,-1]) # residual. Data type: tf.tensor, Size: (1,Ntest_elementx*Ntest_elementy)

                loss_element_1 = tf.reduce_mean(tf.square(Res_NN_element_1)) # loss function. Data type: tf.tensor, Size: (1,1)
                loss_element_2 = tf.reduce_mean(tf.square(Res_NN_element_2)) # loss function. Data type: tf.tensor, Size: (1,1)
                loss_element_3 = tf.reduce_mean(tf.square(Res_NN_element_3)) # loss function. Data type: tf.tensor, Size: (1,1)
                loss_element_4 = tf.reduce_mean(tf.square(Res_NN_element_4)) # loss function. Data type: tf.tensor, Size: (1,1)
                self.varloss_total = self.varloss_total + loss_element_1 + loss_element_2 + loss_element_3
 
        self.lossb = tf.reduce_mean(tf.square(self.u_bound_tensor - self.u_pred_boundary)) + tf.reduce_mean(tf.square(self.v_bound_tensor - self.v_pred_boundary)) # loss function. Data type: tf.tensor, Size: (1,1)
        self.lossv = self.varloss_total # 

        # self.lossp= tf.reduce_mean(tf.square(self.f_pred - self.ftrain))
       #   
        if scheme == 'VPINNs': # 

            self.loss  = 10*self.lossb + self.lossv  

        if scheme == 'PINNs':
            self.loss  = 10*self.lossb + self.lossp 
        
        self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    #    self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init) 



        u_NN_numpy = self.net_u(x_quad_element, y_quad_element)
        d1xu_NN_numpy, d2xu_NN_numpy = self.net_dxu(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
        d1yu_NN_numpy, d2yu_NN_numpy = self.net_dyu(x_quad_element, y_quad_element) # first and second derivatives of the predicted values at the quadrature points. Data type: tf.tensor, Size: (N_quad,1)
        #print shape and type of u_NN_numpy
        

        
###############################################################################
###############################################################################
    def plot_3d(self,x,y,a,file_name):
        

        fig3d = plt.figure()
        ax3D = fig3d.add_subplot(111,projection='3d')
        ax3D.contour3D(x,y,a,50,cmap='jet',levels=500,alpha=0.8)
        ax3D.plot_surface(x,y,a,cmap='jet',edgecolor='black',linewidth=0.75)
        # ax3D.contour(x,y,a,15,cmap='viridis',offset=-0.01)
        fontsize = 14
        ax3D.set_xlabel('$x$' , fontsize = fontsize)
        ax3D.set_ylabel('$y$' , fontsize = fontsize)
        ax3D.set_zlabel('$residue$' , fontsize = fontsize)
        # ax3D.set_zlim(-0.01,0.01)
        # ax3D.view_init(elev=25,azim=250,roll=0,vertical_axis='z')
        ax3D.set_title(''.join(['Galerkin Residue']),fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        fig3d.set_size_inches(w=11,h=11)
        plt.savefig(file_name,dpi=300,bbox_inches='tight')
        plt.close()

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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)
 
    
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
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)[:,0:1]
        return u
    def net_v(self, x, y):  
        v = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)[:,1:2]
        return v
    def net_p(self, x, y):  
        p = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)[:,2:3]
        return p
    
    def net_u_numpy(self, x, y):
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)
        return u.eval(session=self.sess)

    def net_tau(self,x,y):
        tau = 0.21
        # x = x.eval(session=self.sess)
        # y = y.eval(session=self.sess)
        # if np.allclose(x,0) or np.allclose(x,1) or np.allclose(y,0) or np.allclose(y,1):
        #     tau = 0.0
        # elif (0<x<0.02) or (0.98<x<1):
        #     tau = 0.03
        return tau
    
    def net_dxu(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu, d2xu
    
    def net_dxv(self, x, y):
        v   = self.net_v(x, y)
        d1xv = tf.gradients(v, x)[0]
        d2xv = tf.gradients(d1xv, x)[0]
        return d1xv, d2xv
    
    def net_dxu_numpy(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu.eval(session=self.sess), d2xu.eval(session=self.sess)
    

    def net_dyu(self, x, y):
        u   = self.net_u(x, y)
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        return d1yu, d2yu 

    def net_dyv(self, x, y):
        v   = self.net_v(x, y)
        d1yv = tf.gradients(v, y)[0]
        d2yv = tf.gradients(d1yv, y)[0]
        return d1yv, d2yv 
    
    def net_dyu_numpy(self, x, y):
        u   = self.net_u(x, y)
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        return d1yu.eval(session=self.sess), d2yu.eval(session=self.sess)

    def net_f(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        ftemp = d2xu + d2yu
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

    def grad_test_func(self, N_test,x):
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
      

###############################################################################
    def train(self, nIter):
        
        tf_dict = {self.x_bound_tensor: self.x_bound  , self.y_bound_tensor: self.y_bound,
                   self.u_bound_tensor: self.u_bound, self.v_bound_tensor: self.v_bound,
                   self.x_test_tensor: self.x_test, self.y_test_tensor: self.y_test, # 
                   self.x_force_tensor: self.x_f, self.y_force_tensor: self.y_f,
                   self.force_1_tensor: self.f1train, self.force_2_tensor: self.f2train}

        start_time   = time.time()
        for it in track(range(nIter), description='Training: '): # 

            self.sess.run(self.train_op_Adam, tf_dict) 

            loss_value = self.sess.run(self.loss, tf_dict)
            loss_his.append(loss_value)
#            if it % 1 == 0:
#                loss_value = self.sess.run(self.loss, tf_dict)
#                u_pred     = self.sess.run(self.c_test, tf_dict)
#                u_pred_his.append(u_pred)
            if it % 100 == 0:
                elapsed = time.time() - start_time 
                str_print = ''.join(['It: %d, Loss: %.3e, Time: %.2f'])
                print(str_print % (it, loss_value, elapsed))
                start_time = time.time()

    def predict_u(self):
        u_pred = self.sess.run(self.u_test, {self.x_test_tensor: self.x_test, self.y_test_tensor: self.y_test})
        return u_pred
    def predict_v(self):
        v_pred = self.sess.run(self.v_test, {self.x_test_tensor: self.x_test, self.y_test_tensor: self.y_test})
        return v_pred
    def predict_p(self):
        p_pred = self.sess.run(self.p_test, {self.x_test_tensor: self.x_test, self.y_test_tensor: self.y_test})
        return p_pred
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
        N_force = number of residual points in PINNs
    '''
    scheme = 'VPINNs'
    Net_layer = [2] + [20] * 3 + [3] # Network structure

    var_form  = 3
    N_el_x = 1
    N_el_y = 1
    N_test_x = N_el_x*[10]
    N_test_y = N_el_y*[10]
    N_quad = 70
    N_bound = 80
    N_force = 100    
    

    ###########################################################################
    def testfunc_x(n,x):
        '''
        Test function in x direction
        input:
            - n,  Type: int, Size: (1). Order of the jacobi polynomial
            - x,  Type: numpy.ndarray, Size: (N_quad,1). Quadrature points in x direction
        output:
            - test, Type: numpy.ndarray, Size: (N_quad,1). Value of the test function at quadrature points
        '''
        test_x  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
        return test_x
    
    def testfunc_y(n,y):
        '''
        Test function in y direction
        input:
            - n,  Type: int, Size: (1). Order of the jacobi polynomial
            - y,  Type: numpy.ndarray, Size: (N_quad,1). Quadrature points in y direction
        output:
            - test, Type: numpy.ndarray, Size: (N_quad,1). Value of the test function at quadrature points
        '''
        test_y  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
        return test_y
    
    def grad_test_func( N_test,x):
        '''
        Derivative of the test function. Works for Jacobi polynomials.
        input:
            - N_test, Type: int, Size: (1). Order of the derivative of the test function
            - x,      Type: numpy.ndarray, Size: (N_quad,1). Quadrature points in x direction
        output:
            - d1test, Type: numpy.ndarray, Size: (N_quad,1). Value of the derivative of the test function at quadrature points
        '''
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

    def tau(x,y):
        tau = 0.21
        if np.allclose(x,0) or np.allclose(x,1) or np.allclose(y,0) or np.allclose(y,1):
            tau = 0.0
        return tau
    ###########################################################################    
    omegax = 2*np.pi
    omegay = 2*np.pi
    r1 = 10

    def u_bound_ext(x, y):
        '''
        Dirichlet boundary condition
        input: 
            - x,  Type: numpy.ndarray, Size: (N_bound,1)
            - y,  Type: numpy.ndarray, Size: (N_bound,1)
        output: 
            - u_bound, Type: numpy.ndarray, Size: (N_bound,1)
        '''
        u_bound = x*0.0 + y*0.0
        # if all elements of y are 1.0, return 1.0
        u_bound[np.where(np.all(np.isclose(y,1.0),axis=1))] = 1.0
        return u_bound
    def v_bound_ext(x, y):
        '''
        Dirichlet boundary condition
        input: 
            - x,  Type: numpy.ndarray, Size: (N_bound,1)
            - y,  Type: numpy.ndarray, Size: (N_bound,1)
        output: 
            - v_bound, Type: numpy.ndarray, Size: (N_bound,1)
        '''
        v_bound = x*0.0 + y*0.0
        return v_bound
    
    def p_bound_ext(x, y):
        '''
        Dirichlet boundary condition
        input: 
            - x,  Type: numpy.ndarray, Size: (N_bound,1)
            - y,  Type: numpy.ndarray, Size: (N_bound,1)
        output: 
            - p_bound, Type: numpy.ndarray, Size: (N_bound,1)
        '''
        p_bound = x*0.0 + y*0.0
        return p_bound

    def f_1_ext(x,y):
        '''
        Forcing term
        input:
            - x,  Type: numpy.ndarray, Size: (N_force,1)
            - y,  Type: numpy.ndarray, Size: (N_force,1)
        output:
            - f1_temp, Type: numpy.ndarray, Size: (N_force,1)
        '''
        f1_temp = x*0.0 + y*0.0 + 0.0
        return f1_temp
    
    def f_2_ext(x,y):
        '''
        Forcing term
        input:
            - x,  Type: numpy.ndarray, Size: (N_force,1)
            - y,  Type: numpy.ndarray, Size: (N_force,1)
        output:
            - f2_temp, Type: numpy.ndarray, Size: (N_force,1)
        '''
        f2_temp = x*0.0 + y*0.0 + 0.0
        return f2_temp
    
    def u_exact(x, y):
        '''
        Exact Solution
        input: 
            - x,  Type: numpy.ndarray, Size: (N_bound,1)
            - y,  Type: numpy.ndarray, Size: (N_bound,1)
        output: 
            - c_bound, Type: numpy.ndarray, Size: (N_bound,1)
        '''
        u_exact = x*0.0 + y*0.0
        return u_exact
    def v_exact(x,y):
        v_exact = x*0.0 + y*0.0
        return v_exact
    def p_exact(x,y):
        p_exact = x*0.0 + y*0.0
        return p_exact
    ###########################################################################
    # Boundary points
    #For a domain [0,1]^2, use Latin Hypercube Sampling (lhs) to generate boundary points.
    #If domain is [-1,1]^2, use 2*lhs(1,N_bound)-1 (for example, x_top = 2*lhs(1,N_bound)-1)

    x_top = lhs(1,N_bound) # Datatype: numpy.ndarray, Size: (N_bound,1), Sample numbers between 0 to 1
    y_top = np.empty(len(x_top))[:,None]            # Datatype: numpy.ndarray, Size: (N_bound,1)
    
    y_top.fill(1)                                   # Fill the array with 1, to denote top boundary
    coord_train_top = np.hstack((x_top, y_top))     # Datatype: numpy.ndarray, Size: (N_bound,2), Place x and y coordinates together side by side


    u_bound_top = np.empty(len(x_top))[:,None]        # Datatype: numpy.ndarray, Size: (N_bound,1)
    u_bound_top = u_bound_ext(x_top, y_top)                 # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    u_top_train = u_bound_top                         # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    v_bound_top = np.empty(len(x_top))[:,None]        # Datatype: numpy.ndarray, Size: (N_bound,1)
    v_bound_top = v_bound_ext(x_top, y_top)                 # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    v_top_train = v_bound_top                         # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    x_bottom = lhs(1,N_bound)                       # Datatype: numpy.ndarray, Size: (N_bound,1). Sample numbers between 0 to 1
    y_bottom = np.empty(len(x_bottom))[:,None]      # Datatype: numpy.ndarray, Size: (N_bound,1)
    
    y_bottom.fill(0)                                # Fill the array with 0, to denote bottom boundary
    coord_train_bottom = np.hstack((x_bottom, y_bottom)) # Datatype: numpy.ndarray, Size: (N_bound,2), Place x and y coordinates together side by side

    u_bound_bottom = np.empty(len(x_bottom))[:,None]  # Datatype: numpy.ndarray, Size: (N_bound,1)
    u_bound_bottom = u_bound_ext(x_bottom, y_bottom)        # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    u_bottom_train = u_bound_bottom                        # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    v_bound_bottom = np.empty(len(x_bottom))[:,None]  # Datatype: numpy.ndarray, Size: (N_bound,1)
    v_bound_bottom = v_bound_ext(x_bottom, y_bottom)        # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    v_bottom_train = v_bound_bottom                        # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points



    y_right = lhs(1,N_bound)                        # Datatype: numpy.ndarray, Size: (N_bound,1). Sample numbers between 0 to 1
    x_right = np.empty(len(y_right))[:,None]        # Datatype: numpy.ndarray, Size: (N_bound,1)
    
    x_right.fill(1)                                 # Fill the array with 1, to denote right boundary
    coord_train_right = np.hstack((x_right, y_right))  # Datatype: numpy.ndarray, Size: (N_bound,2), Place x and y coordinates together side by side



    u_bound_right = np.empty(len(y_right))[:,None]    # Datatype: numpy.ndarray, Size: (N_bound,1)
    u_bound_right = u_bound_ext(x_right, y_right)           # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    u_right_train = u_bound_right                        # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    v_bound_right = np.empty(len(y_right))[:,None]    # Datatype: numpy.ndarray, Size: (N_bound,1)
    v_bound_right = v_bound_ext(x_right, y_right)           # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    v_right_train = v_bound_right                        # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    y_left = lhs(1,N_bound)                            # Datatype: numpy.ndarray, Size: (N_bound,1). Sample numbers between 0 to 1
    x_left = np.empty(len(y_left))[:,None]             # Datatype: numpy.ndarray, Size: (N_bound,1)
    
    x_left.fill(0)                                     # Fill the array with 0, to denote left boundary
    coord_left_train = np.hstack((x_left, y_left))     # Datatype: numpy.ndarray, Size: (N_bound,2), Place x and y coordinates together side by side


    u_bound_left = np.empty(len(y_left))[:,None]         # Datatype: numpy.ndarray, Size: (N_bound,1)
    u_bound_left = u_bound_ext(x_left, y_left)                 # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    u_left_train = u_bound_left                          # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points

    v_bound_left = np.empty(len(y_left))[:,None]         # Datatype: numpy.ndarray, Size: (N_bound,1)
    v_bound_left = v_bound_ext(x_left, y_left)                 # Evaluate the exact solution at boundary points using given dirichlet boundary condition
    v_left_train = v_bound_left                          # Datatype: numpy.ndarray, Size: (N_bound,1), Place the exact solution at boundary points



    coord_all_train = np.concatenate((coord_train_top, coord_train_bottom, coord_train_right, coord_left_train)) # Datatype: numpy.ndarray, Size: (4*N_bound,2), Place all the coordinates together on top of each other
    u_all_train = np.concatenate((u_top_train, u_bottom_train, u_right_train, u_left_train))                    # Datatype: numpy.ndarray, Size: (4*N_bound,1), Place all the exact solution together on top of each other
    v_all_train = np.concatenate((v_top_train, v_bottom_train, v_right_train, v_left_train))                    # Datatype: numpy.ndarray, Size: (4*N_bound,1), Place all the exact solution together on top of each other


    ###########################################################################
    # Sampling points for forcing term in the PDE
    # Forcing is a vector function for Stokes equation
    #For a domain [0,1]^2, use Latin Hypercube Sampling (lhs) to generate boundary points.
    #If domain is [-1,1]^2, use 2*lhs(1,N_bound)-1 (for example, x_top = 2*lhs(1,N_bound)-1)

    coord_train_force = lhs(2,N_force)                        # Datatype: numpy.ndarray, Size: (N_force,2), Sample numbers between 0 to 1 for x and y coordinates of residual points

    x_force = coord_train_force[:,0]                          # Datatype: numpy.ndarray, Size: (N_force,1), x coordinates of force points
    y_force = coord_train_force[:,1]                          # Datatype: numpy.ndarray, Size: (N_force,1), y coordinates of force points

    f_1_train = f_1_ext(x_force,y_force)[:,None] # Datatype: numpy.ndarray, Size: (N_force,1), Evaluate the forcing term at force points
    f_2_train = f_2_ext(x_force,y_force)[:,None] # Datatype: numpy.ndarray, Size: (N_force,1), Evaluate the forcing term at force points



    ###########################################################################
    # Quadrature points
    # Quadrature points are used to evaluate the integral in the variational formulation
    [quad_x_coord, quad_x_weight] = GaussLobattoJacobiWeights(N_quad, 0, 0) # Datatype: numpy.ndarray, Size: (N_quad,), Gauss-Lobatto-Jacobi quadrature points and weights in x direction
    quad_y_coord, quad_y_weight   = (quad_x_coord, quad_x_weight) # Datatype: numpy.ndarray, Size: (N_quad,), Gauss-Lobatto-Jacobi quadrature points and weights in y direction

    # Create meshgrid of quadrature points 
    quad_x_coord_mesh, quad_y_coord_mesh                   = np.meshgrid(quad_x_coord,  quad_y_coord)
    # Create meshgrid of quadrature weights
    quad_x_weight_mesh, quad_y_weight_mesh                 = np.meshgrid(quad_x_weight, quad_y_weight)
    # Aggregrate all quadrature points and their coordiantes
    quad_coord_train                                          = np.hstack((quad_x_coord_mesh.flatten()[:,None],  
    quad_y_coord_mesh.flatten()[:,None]))
    # Aggregrate all quadrature weights
    quad_weight_train                                         = np.hstack((quad_x_weight_mesh.flatten()[:,None], quad_y_weight_mesh.flatten()[:,None])) # Datatype: numpy.ndarray, Size: (N_quad*N_quad,2), Quadrature weights in x and y direction

    ###########################################################################
    # Construction of RHS for VPINNs
    NE_x, NE_y = N_el_x, N_el_y

    [x_left, x_right] = [0, 1] # Domain boundaries in x direction. Datatype: float
    [y_bottom, y_up] = [0, 1]

    delta_x    = (x_right - x_left)/NE_x
    delta_y    = (y_up - y_bottom)/NE_y

    grid_x     = np.asarray([ x_left + i*delta_x for i in range(NE_x+1)])   # Datatype: numpy.ndarray, Size: (N_el_x+1,), x coordinates of grid points
    grid_y     = np.asarray([ y_bottom + i*delta_y for i in range(NE_y+1)]) # Datatype: numpy.ndarray, Size: (N_el_y+1,), y coordinates of grid points

    # num_total_testfunc = [(len(grid_x)-1)*[N_test_x], (len(grid_y)-1)*[N_test_y]] # Datatype: list, Size: (2, N_el_x), Number of test functions in each element
    num_total_testfunc = [N_test_x, N_test_y]
 
    #+++++++++++++++++++
    x_quad  = quad_coord_train[:,0:1] # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), x coordinates of quadrature points
    y_quad  = quad_coord_train[:,1:2] # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), y coordinates of quadrature points
    w_quad  = quad_weight_train       # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), quadrature weights

    u_bound_total = []                  # 
    v_bound_total = []                  # 
    F_1_ext_total = []
    F_2_ext_total = []

    NE_x, NE_y = N_el_x, N_el_y

    for ex in range(NE_x):         # Loop over elements in x direction
        for ey in range(NE_y):     # Loop over elements in y direction
            num_testfunc_elem_x  = num_total_testfunc[0][ex] # Number of test functions in current element in x direction, Datatype: int
            num_testfunc_elem_y  = num_total_testfunc[1][ey] # Number of test functions in current element in y direction, Datatype: int
            
            # Standard quadrature points are defined for [-1,1]^2. This transformation is to map the quadrature points to the current element.
            x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1) # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), x coordinates of quadrature points in current element

            y_quad_element = grid_y[ey] + (grid_y[ey+1]-grid_y[ey])/2*(y_quad+1) # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), y coordinates of quadrature points in current element

            # Reference element has a side of length 2. This transformation is to account for the lengths and area of the current element with respect to the reference element. 
            jacobian_x     = ((grid_x[ex+1]-grid_x[ex])/2) # Datatype: float, Size: (1,), Jacobian in x direction
            jacobian_y     = ((grid_y[ey+1]-grid_y[ey])/2) # Datatype: float, Size: (1,), Jacobian in y direction
            jacobian       = ((grid_x[ex+1]-grid_x[ex])/2)*((grid_y[ey+1]-grid_y[ey])/2) # Datatype: float, Size: (1,), Jacobian in x and y direction

            
            test_x_quad_element = np.asarray([ testfunc_x(n,x_quad)  for n in range(1, num_testfunc_elem_x+1)]) # Datatype: numpy.ndarray, Size: (N_test_x, N_quad*N_quad), Value of test functions in x direction at quadrature points in current element
            test_y_quad_element = np.asarray([ testfunc_y(n,y_quad)  for n in range(1, num_testfunc_elem_y+1)]) # Dataype: numpy.ndarray, Size: (N_test_y, N_quad*N_quad), Value of test functions in y direction at quadrature points in current element
    
            ddx_test_x_quad_element = grad_test_func(num_testfunc_elem_x, x_quad) # Datatype: numpy.ndarray, Size: (N_test_x, N_quad*N_quad), Value of derivative of test functions in x direction at quadrature points in current element

            ddy_test_y_quad_element = grad_test_func(num_testfunc_elem_y, y_quad) # Datatype: numpy.ndarray, Size: (N_test_y, N_quad*N_quad), Value of derivative of test functions in y direction at quadrature points in current element

            c_quad_element = f_1_ext(x_quad_element, y_quad_element) # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), Value of exact solution at quadrature points in current element
            f_1_quad_element = f_1_ext(x_quad_element, y_quad_element)     # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), Value of forcing term at quadrature points in current element
            f_2_quad_element = f_2_ext(x_quad_element, y_quad_element)     # Datatype: numpy.ndarray, Size: (N_quad*N_quad,1), Value of forcing term at quadrature points in current element



            
            tau = 0.1
            b_x = 1.0
            b_y = 0.0

            
            # \int_{\Omega} c v dx dy = \int_{\Omega} c * v_x * v_y dx dy
            # Multiply by jacobian to account for the area of the current element. Convert integral into summation over quadrature points
            # Jacobian*\sum_{i=1}^{N_quad} w_x_i * w_y_i * c(i) * v_x_i * v_y_i
            # For each test function in x and y direction
            # Hence, resulting size is (N_test_x, N_test_y)
            c_bound_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*test_x_quad_element[r]*w_quad[:,1:2]*test_y_quad_element[k]*c_quad_element) \
                            for r in range(num_testfunc_elem_x)] for k in range(num_testfunc_elem_y)]) # This term can be ignored. It is used to check the accuracy of the solution or for inverse problems. Exact solutions are only defined on boundaries and not in the domain.
    
            # \int_{\Omega} f v dx dy = \int_{\Omega} f * v_x * v_y dx dy
            # Multiply by jacobian to account for the area of the current element. Convert integral into summation over quadrature points
            # Jacobian*\sum_{i=1}^{N_quad} w_x_i * w_y_i * f(i) * v_x_i * v_y_i
            # For each test function in x and y direction
            # Hence, resulting size is (N_test_x, N_test_y)
            F_1_ext_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*test_x_quad_element[r]*w_quad[:,1:2]*test_y_quad_element[k]*f_1_quad_element) \
                            for r in range(num_testfunc_elem_x)] for k in range(num_testfunc_elem_y)])
            F_2_ext_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*test_x_quad_element[r]*w_quad[:,1:2]*test_y_quad_element[k]*f_2_quad_element) \
                            for r in range(num_testfunc_elem_x)] for k in range(num_testfunc_elem_y)])
            
                
            
                
            # c_bound_total.append(c_bound_element)
    
            F_1_ext_total.append(F_1_ext_element) 
            F_2_ext_total.append(F_2_ext_element)
    
#    U_ext_total = np.reshape(U_ext_total, [N_el_x, N_el_y, N_test_y, N_test_x])
    F_1_ext_total = np.reshape(F_1_ext_total, [N_el_x, N_el_y, N_test_y[0], N_test_x[0]]) # Datatype: numpy.ndarray, Size: (N_el_x, N_el_y, N_test_y, N_test_x), RHS for VPINNs
    F_2_ext_total = np.reshape(F_2_ext_total, [N_el_x, N_el_y, N_test_y[0], N_test_x[0]]) # Datatype: numpy.ndarray, Size: (N_el_x, N_el_y, N_test_y, N_test_x), RHS for VPINNs
    
    ###########################################################################
    # Test points
    delta_test = 0.01
    xtest = np.arange(x_left, x_right + delta_test, delta_test) # Datatype: numpy.ndarray, Size: (N_test,), x coordinates of test points
    ytest = np.arange(y_bottom, y_up + delta_test, delta_test) # Datatype: numpy.ndarray, Size: (N_test,), y coordinates of test points

    U_test = np.asarray([[ [xtest[i],ytest[j],u_exact(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))]) # Datatype: numpy.ndarray, Size: (N_test, N_test, 3), x and y coordinates of test points and the exact solution at test points
    x_test = U_test.flatten()[0::3] # Datatype: numpy.ndarray, Size: (N_test*N_test,), x coordinates of test points
    y_test = U_test.flatten()[1::3] # Datatype: numpy.ndarray, Size: (N_test*N_test,), y coordinates of test points
    exact_u = U_test.flatten()[2::3] # Datatype: numpy.ndarray, Size: (N_test*N_test,), exact solution at test points
    coord_test = np.hstack((x_test[:,None],y_test[:,None]))
    u_test = exact_u[:,None]

    V_test = np.asarray([[ [xtest[i],ytest[j],v_exact(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))]) # Datatype: numpy.ndarray, Size: (N_test, N_test, 3), x and y coordinates of test points and the exact solution at test points
    exact_v = V_test.flatten()[2::3] # Datatype: numpy.ndarray, Size: (N_test*N_test,), exact solution at test points
    v_test = exact_v[:,None]

    P_test = np.asarray([[ [xtest[i],ytest[j],p_exact(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))]) # Datatype: numpy.ndarray, Size: (N_test, N_test, 3), x and y coordinates of test points and the exact solution at test points
    exact_p = P_test.flatten()[2::3] # Datatype: numpy.ndarray, Size: (N_test*N_test,), exact solution at test points
    p_test = exact_p[:,None]




    ###########################################################################
    model = VPINN(coord_all_train, u_all_train,v_all_train, coord_train_force, f_1_train, f_2_train, quad_coord_train, quad_weight_train,\
                 F_1_ext_total, F_2_ext_total, grid_x, grid_y, num_total_testfunc, coord_test, u_test,v_test,p_test, Net_layer)
    
    vec_pred_his, loss_his = [], []
    model.train(50000 + 1)
    u_pred = model.predict_u()
    v_pred = model.predict_v()
    p_pred = model.predict_p()

#%%
    ###########################################################################
    # =============================================================================
    #    Plotting
    # =============================================================================
  
    fontsize = 24
    fig = plt.figure(1)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(loss_his)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_loss','.pdf']))
    
    ###########################################################################
    x_train_plot, y_train_plot = zip(*coord_all_train)
    x_f_plot, y_f_plot = zip(*coord_train_force)
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
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel('$x$', fontsize = fontsize)
    plt.ylabel('$y$', fontsize = fontsize)
    #ax.set_aspect(1)
    ax.locator_params(nbins=5)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Domain','.pdf']))
    
    ###########################################################################
    x_test_plot = np.asarray(np.split(coord_test[:,0:1].flatten(),len(ytest)))
    y_test_plot = np.asarray(np.split(coord_test[:,1:2].flatten(),len(ytest)))
    u_pred_plot = np.asarray(np.split(u_pred.flatten(),len(ytest)))
    v_pred_plot = np.asarray(np.split(v_pred.flatten(),len(ytest)))
    p_pred_plot = np.asarray(np.split(p_pred.flatten(),len(ytest)))

    
    
    fontsize = 32
    labelsize = 26
   
    
    
    
    fig_pred_u, ax_pred_u = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred_u.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred_u.colorbar(CS_pred, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred_u.locator_params(nbins=8)
    ax_pred_u.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred_u.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred_u.set_aspect(1)
    #fig.tight_layout()
    fig_pred_u.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_U_',scheme,'_Predict','.png']))
    
    fig_pred_v, ax_pred_v = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred_v.contourf(x_test_plot, y_test_plot, v_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred_v.colorbar(CS_pred, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred_v.locator_params(nbins=8)
    ax_pred_v.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred_v.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred_v.set_aspect(1)
    #fig.tight_layout()
    fig_pred_v.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_V_',scheme,'_Predict','.png']))

    fig_pred_p, ax_pred_p = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred_p.contourf(x_test_plot, y_test_plot, p_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred_p.colorbar(CS_pred, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred_p.locator_params(nbins=8)
    ax_pred_p.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred_p.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred_p.set_aspect(1)
    #fig.tight_layout()
    fig_pred_p.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_P_',scheme,'_Predict','.png']))
    
   
    
    
        