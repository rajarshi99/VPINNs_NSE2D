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