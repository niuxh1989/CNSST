#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering

from sympy.utilities.iterables import multiset_permutations
import numpy as np

class CNSST:
    def __init__(self, X, adjuster = 90, sensitivity = 0, data_instance = None, n_clusters = None, original_constants = np.array([ 1, 1, 1]).astype(float), weights_ = np.array([1, 1, 1]).astype(float), sort_col = False, i_column = None, seed_ = 1):

        """Perform CNSST clustering from input array X

        CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering

        Parameters
        -----------
        X: array, look at example, shape can be (a, b, c), (a, b)
        adjuster: default = 90, int
        sensitivity:  default = 0, int
        data_instance: for assessing feature importances
        n_clusters: number of clusters, int
        original_constants: vector v for views' construction, ndarray
        weights_: w, ndarray
        sort_col: flag to determine whether to sort column
        i_column: index of column to be sorted
        seed_ : seed for sorting column
        """
        # change negative number
        for i in range(0, len(X[0])):
            if np.any(X[:,i]<0) == True:
                X[:,i] = X[:,i] + np.min(X[:,i])*(-1)
      
        self.adjuster = adjuster
        self.sensitivity = sensitivity
        self.o_c = 0
        self.weights__ = 0
        self.o_c_stor_ = original_constants
        self.weights__stor_ = weights_

        self.o_c_stor_init__ = np.array([1, 1, 1]).astype(float)
        self.weights_init__ = np.array([1, 1, 1]).astype(float)
        #dec 0.55, dec 24
        if len(X.shape) == 2:
            X = transform_2(X)
        y_axis, x_axis, dim = X.shape
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_start = 0
        self.x_end = x_axis
        self.y_start = 0
        self.y_end = y_axis

        self.X = X
        self.array_1 = (X/adjuster).astype(float).copy() # approximation
        self.array = position_index(self.array_1).copy()
        
        test = self.array[:,2:(len(self.array[0]))]
        # calculate feature inmportances from chosen sample (70%)   sort_column = False, s_c = None, seed_s = 1
        #----------------------------------------------------------
        self._importances = get_importances(test, data_instance,  sort_column = sort_col, s_c = i_column, seed_s = seed_)
        #----------------------------------------------------------
        # only record data
        self._data_ = data_instance.data
        self.array[:,2:(len(self.array[0]))] = rearrange_col(self._importances, test, ord_ = 'dec')
        self.assign_c = len(np.where(self._importances - np.mean(self._importances)>0)[0])
        self.adjuster = adjuster
        

        # for supervised training
        self.data_instance = data_instance
        self.n_clusters = n_clusters
#         # initiate
#         self.n_ = n_p
        # features' influence
        self.impact_record = []

        self.flag_explore = False
        global original_arr_
        original_arr_ = self.array
         
    def cluster(self):#---------------------------------------------------------------------------------------
        self.threshold = float(self.Y_increment_function_2[self.threshold_index])
        self.clusters=[[0]]
        for i in range(1, (len(self.Y_increment_function_1)-1)):
            if (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) > 0:
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) < 0 and (self.Y_increment_function_1[i-2]-self.threshold)*(self.Y_increment_function_1[i-1]-self.threshold) > 0 : # added
                self.clusters.append([])
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) < 0 and (self.Y_increment_function_1[i-2]-self.threshold)*(self.Y_increment_function_1[i-1]-self.threshold)<= 0: # added
                self.clusters[len(self.clusters)-1].append(i) #
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) > 0:
                self.clusters.append([]) # added
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) < 0:
                self.clusters.append([])
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) == 0:
                self.clusters[len(self.clusters)-1].append(i)

        # last point: (added)
        if (self.Y_increment_function_1[len(self.Y_increment_function_1)-1]-self.threshold)*(self.Y_increment_function_1[len(self.Y_increment_function_1)-2]-self.threshold) >= 0:  # last point clustering
            self.clusters[len(self.clusters)-1].append(len(self.Y_increment_function_1)-1)
        elif (self.Y_increment_function_1[len(self.Y_increment_function_1)-1]-self.threshold)*(self.Y_increment_function_1[len(self.Y_increment_function_1)-2]-self.threshold)< 0:
            self.clusters.append([])
            self.clusters[len(self.clusters)-1].append(len(self.Y_increment_function_1)-1)


    def create_label(self):
        self.labels = np.zeros((self.y_axis,self.x_axis))
        self.x = rearrange(self.F, self.array[:,0]).astype(int) #---------------------------------------------------------------
        self.y = rearrange(self.F, self.array[:,1]).astype(int) #---------------------------------------------------------------
        for m in range(0,len(self.clusters)):
            for n in range(0,len(self.clusters[m])):
                self.labels[self.x[self.clusters[m][n]],self.y[self.clusters[m][n]]] = m
        h, w = self.labels.shape
        self._labels = np.reshape(self.labels, (h*w))

    # need to improve
    def __F_function(self, array_):
        a_3 = array_
        h, w = a_3.shape
        d = w -2
        
        def initial_alpha(d, arg):
            '''initial alpha
            paremeters:
            -----------------
            d: dimension
            arg: original sequence, here is np.array([1, 1.22, 1.28])
            return:
            -----------------
            matrix '''
            alpha_vector = np.ones(((d, d)),dtype=float)
            a = generate_constants(d, arg)
            for i in range(0,self.assign_c): 
                alpha_vector[:,i] = a
            return alpha_vector

        def generate_constants(d,arg):
            '''
            parameters:
            ----------------------------------
            d:dimension
            arg:original sequence, here is np.array([1, 1.22, 1.28])

            return:
            ----------------------------------
            a generated sequence based on np.array([1, 1.22, 1.28])
               '''
            n = d - 3
            B = arg.astype(float) # changed------------------------------------------------------------------------------------
            B_1 = B.copy()
            count_position = np.array([len(B)-2,len(B)])
            counter = 0
            while counter < n:
                if count_position[0] >= 0:
                    number_for_inserting = np.mean(B[count_position[0]:count_position[1]])
                    B_1 = np.insert(B_1, count_position[1]-1, number_for_inserting)
                    count_position = count_position - np.array([1,1])
                else:
                    B = B_1
                    B_1 = B.copy()
                    count_position = np.array([len(B)-2,len(B)])
                    counter = counter - 1
                counter = counter + 1
            return B_1


        def transform_1_0(arg):
            if len(arg.shape) != 1:
                raise ValueError("arg shape must be (a, )")
            h = arg.shape[0]
            w = 1
            d = 1
            return np.reshape(arg, (d, h*w))

        # x data matrix
        x = a_3[:,2:2+d]
        original_constants = self.o_c
        weights_ = self.weights__

        # b: alfa_vector
        b = initial_alpha(d, original_constants)


        def r_r_product(row_, matrix_):
            '''new matrix by mutiplying row_ with every row of matrix
            parameters:
            -----------------
            row_: array, shape (a,)
            matrix_: array, shape (a, b)

            return:
            -----------------
            matrix: new matrix by mutiplying row_ with every row of matrix
                   array, shape(a, b) '''
            matrix = matrix_.copy()
            for i in range(0, len(matrix)):
                matrix[i] = row_*matrix[i]
            return matrix

        r = []
        for i in range(0, len(x)):
            r.append(r_r_product(x[i], b))
        r = np.array(r)
        _e =np.ones(r.shape).astype(float)*(0.01)
        basic_matrix = ((r + _e).astype(np.cdouble))**5.7
        zeroes = np.zeros(x.shape).astype(float)

        
        # sq_sum_matrix
        sq_sum_matrix = np.zeros(x.shape).astype(float)
        h, w, d = basic_matrix.shape
        for i_r in range(0,h):
            for i_c in range(0,w):
                const_ = np.sum(basic_matrix[i_r][0][1:d].astype(np.cdouble))
                variable = basic_matrix[i_r][i_c,0].astype(np.cdouble)
                sq_sum_matrix[i_r,i_c] = const_+variable
    
        sq_sum_matrix = sq_sum_matrix.astype(np.cdouble)
                
        # two_sq_matrix
        two_sq_matrix = np.zeros(x.shape).astype(float)
        for i in range(0,h):
            two_sq_matrix[i] = basic_matrix[i,0]
        # change type
        two_sq_matrix = two_sq_matrix.astype(np.cdouble)
        

        # _sq_matrix
        _sq_matrix = np.zeros(x.shape).astype(float)
        for i in range(0,h):
            _sq_matrix[i] = basic_matrix[i,1]
        _sq_matrix = np.insert(_sq_matrix, 0, _sq_matrix[:,len(_sq_matrix[0])-1], axis = 1 )
        _sq_matrix = np.delete(_sq_matrix, len(_sq_matrix[0])-1, axis = 1 )
        # change type
        _sq_matrix = _sq_matrix.astype(np.cdouble)
        

        def arrange_sum_col(func): # f_
            def inner(*args):
                '''
                args[0]: sq_sum_matrix
                args[1]: two_sq_matrix
                args[2]: _sq_matrix
                func[0]: <function four_parts_.<locals>.first_ at 0x0000020BB18E31E0>
                func[1]: <function four_parts_.<locals>.second_ at 0x0000020BB18E3AE8>
                func[2]: <function four_parts_.<locals>.third_ at 0x0000020BB18E3268>
                func[3]: <function four_parts_.<locals>.forth_ at 0x0000020BB18E3D08>

                '''
                h_, w_ = x.shape
                weights = generate_constants(w_, weights_)
                weights = transform_1_0(weights)
                weights = np.repeat(weights, [h_], axis=0)
                f_ = np.zeros(x.shape).astype(float)
                f_ = func[0](args[0],args[1])*func[1](args[0],args[1],args[2])*func[2](args[1])*func[3](args[2])
                f_ = f_.real
                f_ready_for_sum = (((f_*weights).astype(np.cdouble))**2.6)*4
                F = np.sum(f_ready_for_sum, axis = 1)
                return F.real
            return inner

        def four_parts_(func): # four parts
            def first_(sq_sum_n,two_sq_n):
                '''
                calculate (sq_sum_n-two_sq_n)/sq_sum_n for all data vectors
                '''
                result_first_ = (sq_sum_n-two_sq_n)/sq_sum_n
                return result_first_

            def second_(sq_sum_n, two_sq_n,_sq_n):
                '''
                calculate (two_sq_n/(sq_sum_n-_sq_n +1))**(1/5.31) for all data vectors
                '''
                result_second_ = (two_sq_n/(sq_sum_n-_sq_n +1))**(1/5.31)
                return result_second_

            def third_(two_sq_n):
                '''
                calculate (two_sq_n)**(1/15.41) for all data vectors
                '''
                result_third_ = (two_sq_n)**(1/15.41)
                return result_third_

            def forth_(_sq_n):
                '''
                calculate _sq_n**(1/10.31) for all data vectors
                '''
                result_forth_ = _sq_n**(1/10.31)
                return result_forth_

            return first_, second_, third_, forth_

        @arrange_sum_col
        @four_parts_
        def col_(*data): # input data
            return data
        
        F = col_(sq_sum_matrix, two_sq_matrix, _sq_matrix)
        self.F = F.real    
        
    def diff(self):
        self.Y_increment_function_1 = [(lambda a, b  : a-b)(self.F_sorted[i], self.F_sorted[i-1]) for i in range(1,len(self.F_sorted))]
        self.Y_increment_function_1.insert(0,0)

        # preparation for supervised training
        self.recor__ = []
        for num in self.Y_increment_function_1:
            if num != 0:
                self.recor__.append(num)
        self.recor__ = np.array(self.recor__)
             
    def sorting(self):
        self.F_sorted = np.sort(self.F, axis = None)

    def thresh_(self): 
        self.Y_increment_function_2 = sorted(self.Y_increment_function_1).copy()
        for i in range(0, len(self.Y_increment_function_2)):
            if self.Y_increment_function_2[i]>0:
                self.threshold_index = round(i+((len(self.Y_increment_function_2)-i)*self.sensitivity/100))  # adjust sensitivity
                break
   
    def run(self):
        self.__F_function(self.array)
        self.sorting()
        self.diff()
        self.thresh_()
        self.cluster()
        self.create_label()
        return self


class supervised_CNSST(CNSST):
    
    def prepare_(self):
        '''set self.impact_record by using self._importances
           adjust self.array by using self.impact_record'''
        self.significance_f()
        return self

    def set__par_1(self):
        self.o_c = self.o_c_stor_
        self.weights__ = self.weights__stor_

    def run_supervised_1(self):
        '''get self.recor__ and start fit_threshold (arr_1, arr_2, arr_3... ) '''
        # run unsupervised learning, get self.recor__
        self.set__par_1()
        self._CNSST__F_function(self.array)
#         self.sorting()  #change order of target _ 1 according to self.F
        self.sorting()
        self.diff()
        # to start fit_threshold
        return self.adjust_sensitivity()

    def significance_f(self): 
        '''set self.impact_record by using self._importances
           adjust self.array by using self.impact_record'''

        h, w = self.array.shape
        w = w -2

        self._im_record_total_ = []
        # set self.impact_record by using self._importances
        self.impact_record = np.array(sorted(self._importances, reverse =True))*500
     
        _dict__ = dict(zip(np.array([i for i in range(0,w)]), self.impact_record))
        self.dic_for_test = _dict__
     
        # assign weight
        for i in range(0, w):
            self.array[:,i+2] = self.array[:,i+2]*_dict__[i]

    def adjust_sensitivity(self):
        return  self.fit_threshold()

    def fit_threshold(self): 
        '''
        to record:
        original_labels = np.array(clustering._original__.copy())
        arr_1 = clustering.recor__
        arr_2 = clustering.labels_record__
        arr_3 = original_labels
        arr_4 = np.array(clustering._record_dis_diff_sum.copy())
        arr_5 = np.array(clustering._record_dis_diff_mean.copy())
        arr_6 = np.array(clustering._record_dis_diff_m_v.copy())
        
        '''
        self.find_ = []
        
        self.objective_record = []
        self.labels_record__ = []
        
        # added 
        self.clusters_record = []

        self.flag_explore = True
        self._record_dis_diff_m_v = []
        self._record_dis_diff_sum = []
        self._record_dis_diff_mean = []
        self._original__ = []
        self._rdd = []
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.recor__ = self.recor__[np.where(self.recor__ != np.max(self.recor__))]
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        for thres_ in self.recor__:
            self.threshold = thres_
            try:
                self.run_supervised_2()
            except ValueError:
                pass

            c = self.d_map.copy()
            self._original__.append(c)
            self.d_map = self.transform_labels___(self.d_map.copy(), self.n_clusters)
            self.labels_record__.append(self.d_map)
            
       
        self.labels_record__ = np.array(self.labels_record__)
        self.flag_explore = False  
        return self


    def run_supervised_2(self):
        '''run clustering inside fit_threshold (arr_1, arr_2, arr_3... ) with predetermined threshold
           get clusters
           get conrresponding d'''
        self.cluster_supervised()
        self.clusters_record.append(self.clusters)
        self.decision_map()
        return self
    
        
    def cluster_supervised(self):#---------------------------------------------------------------------------------------
        self.clusters=[[0]]
        for i in range(1, (len(self.Y_increment_function_1)-1)):
            if (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) > 0:
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) < 0 and (self.Y_increment_function_1[i-2]-self.threshold)*(self.Y_increment_function_1[i-1]-self.threshold) > 0 : # added
                self.clusters.append([])
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) < 0 and (self.Y_increment_function_1[i-2]-self.threshold)*(self.Y_increment_function_1[i-1]-self.threshold)<= 0: # added
                self.clusters[len(self.clusters)-1].append(i) #
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) > 0:
                self.clusters.append([]) # added
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) < 0:
                self.clusters.append([])
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) == 0:
                self.clusters[len(self.clusters)-1].append(i)

        # last point: (added)
        if (self.Y_increment_function_1[len(self.Y_increment_function_1)-1]-self.threshold)*(self.Y_increment_function_1[len(self.Y_increment_function_1)-2]-self.threshold) >= 0:  # last point clustering
            self.clusters[len(self.clusters)-1].append(len(self.Y_increment_function_1)-1)
        elif (self.Y_increment_function_1[len(self.Y_increment_function_1)-1]-self.threshold)*(self.Y_increment_function_1[len(self.Y_increment_function_1)-2]-self.threshold)< 0:
            self.clusters.append([])
            self.clusters[len(self.clusters)-1].append(len(self.Y_increment_function_1)-1)        
            
    def decision_map(self):
        '''self.clusters -> self.d_map'''
        d_m = self.clusters.copy()
        d_m_r = []
        for i in range(0, len(d_m)):
            for times in range(0, len(d_m[i])):
                d_m_r.append(i)
        self.d_map = np.array(d_m_r)

    def d_map_to_clusters(self):
        '''self.d_map -> self.clusters'''
        t = self.d_map.copy()
        clusters = []
        init_c = 0
        for e in set(t):
            init_v = len(t[np.where(t == e)])
            w = [i for i in range(init_c, init_c + init_v)]
            clusters.append(w)
            init_c = init_c + init_v
        self.clusters = clusters

    def create_label_sup(self):
        self.labels = np.zeros((self.y_axis,self.x_axis))
        self.x = rearrange(self.F, self.array_test[:,0]).astype(int) #---------------------------------------------------------------
        self.y = rearrange(self.F, self.array_test[:,1]).astype(int) #---------------------------------------------------------------
        for m in range(0,len(self.clusters)):
            for n in range(0,len(self.clusters[m])):
                self.labels[self.x[self.clusters[m][n]],self.y[self.clusters[m][n]]] = m
        h, w = self.labels.shape
        self._labels = np.reshape(self.labels, (h*w))
        return self
   
    def transform_labels___(self, _labels__, n_clusters):
        if len(set(_labels__)) == 1:
            if self.flag_explore == True:
                self._rdd.append(None)
                self._record_dis_diff_sum.append(None)
                self._record_dis_diff_mean.append(None)
                self._record_dis_diff_m_v.append(None)
            pass
        else:
            re_arr_, re_arr_counts_ = np.unique(_labels__, return_counts=True)
            ex_ = dict(zip(re_arr_counts_, re_arr_))
            # mean value of positions of all clusters
            record_dis = []
            for i in range(0,len(set(_labels__))):
                record_dis.append(np.mean(np.where(_labels__==i)))
            # absolute value of distance between clusters
            record_dis_diff = [(lambda a, b  : abs(a-b))(record_dis[i], record_dis[i-1]) for i in range(1,len(record_dis))]
           
            if self.flag_explore == True:
                self._rdd.append(record_dis_diff)
                record_dis_diff_sum = np.array(record_dis_diff)
                self._record_dis_diff_sum.append(np.sum(record_dis_diff_sum))
                self._record_dis_diff_mean.append(np.mean(record_dis_diff_sum))
                self._record_dis_diff_m_v.append(np.var(record_dis_diff_sum)/np.mean(record_dis_diff_sum))
            # biggest absolute values of distance according to number of clusters
            number_of_abs_value_dis = n_clusters -1
            max_v_abs_value_dis = []
            record_dis_diff_arr_ = np.array(record_dis_diff)
            for i in range(0,number_of_abs_value_dis):
                max_v_abs_value_dis.append(np.max(record_dis_diff_arr_))
                record_dis_diff_arr_ = np.delete(record_dis_diff_arr_, np.where(record_dis_diff_arr_==np.max(record_dis_diff_arr_)), axis = None)
                if len(record_dis_diff_arr_) == 0:
                    break
            record_dis_diff_arr_ = np.array(record_dis_diff)
            dic_dis_diff_to_ind = dict(zip(record_dis_diff_arr_, np.array([i+0.5 for i in range(0, len(set(_labels__))-1)])))
            b_clusters = []
            for i in max_v_abs_value_dis:
                b_clusters.append(dic_dis_diff_to_ind[i]+0.5)
            b_clusters = np.array(sorted(b_clusters, reverse = False)).astype(int)
            b_clusters = np.insert(b_clusters, 0, 0, axis = None)
            b_clusters = np.append(b_clusters, len(set(_labels__)), axis = None)

            merged_clusters = []
            for i in range(0, len(b_clusters)-1):
                merged_clusters.append([c for c in range(int(b_clusters[i]), b_clusters[i+1])])
            s_m = merged_clusters
            _target_w_ = [i for i in range(0, len(set(_labels__)))]
            _target_x_ = _target_w_.copy()
            for i in range(0, len(merged_clusters)):
                for m in range(0, len(s_m[i])):
                    _target_w_[s_m[i][m]] = i
            ex_dict__ = dict(zip(np.array(_target_x_), np.array(_target_w_)))
            for ind in range(0,len(_labels__)):
                _labels__[ind] = ex_dict__[_labels__[ind]]
        return _labels__
    
    @staticmethod
    def predict(inc): 
        # fisrt seletction by using min and max of (m_v/_mean)/_sum
        
        ######################
        # for stroring inc._labels
        global rec
        # for storing variance of decision map
        global var_original_map
        # for storing variance of decision map by labels
        global var_original_map_uni_
        # for storing m_v
        global rec_arr_6
        # for storing sum
        global rec_arr_4
        # for storing mean
        global rec_arr_5
        # for storing sum of distances 
        global rec_distances_to_centers
        ######################
        rec  = []
        rec_clusters = []
        rec_arr_4 = []
        rec_arr_5 = []
        # added
        var_original_map = []
        var_original_map_uni_ = []
        rec_arr_6 = []
        rec_distances_to_centers = []        
       
        # step 1: encoding process
        #------------------------------------------------encoding process------------------------------------------
        # loop 1: increase weight on features
        for _w_ in range(0,10):
            inc.prepare_()
            par = np.array([0.5, 6.305, 23.6])
            a = multiset_permutations(par) 
            counter_0 = 0
            # loop 2: permutaion of par = np.array([0.5, 6.305, 23.6])
            for i in range(0, 6): 
                cons = np.array(next(a)) 
                inc.o_c_stor_ = cons
                inc.o_c_stor_init__ = cons
                inc.run_supervised_1()
                # plan A
                # arr_4 -> threshold(max clustering._record_dis_diff_sum)
                original_labels = np.array(inc._original__.copy()) #(contain original d)  
                arr_1 = inc.recor__.copy()  #(contain thresholds > 0, decides all the order of calculated results)
                arr_2 = inc.labels_record__.copy() #(contain d')
                arr_3 = original_labels.copy()
                arr_4 = np.array(inc._record_dis_diff_sum.copy()) # (feature of d: sum)
                arr_5 = np.array(inc._record_dis_diff_mean.copy()) # (feature of d: mean)
                arr_6 = np.array(inc._record_dis_diff_m_v.copy()) # (feature of d: mean/variance)

                n_clusters__ = []
                for i in inc.clusters_record:
                    n_clusters__.append(len(i))
                arr_8 = np.array(n_clusters__)

                # criterion
                arr_y_1 = (arr_6/arr_5)/arr_4

                # find threshold corresponding to max sum
                thres_max_sum = arr_1[np.where(arr_5==np.max(arr_5))][0]
                #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                # basic condition, remove None
                ind = np.where(np.array(inc._record_dis_diff_sum) != None)
                arr_1 = arr_1[ind]
                arr_2 = arr_2[ind]
                arr_3 = arr_3[ind]
                arr_4 = arr_4[ind]
                arr_5 = arr_5[ind]
                arr_6 = arr_6[ind]
                arr_8 = arr_8[ind]
                arr_y_1 = arr_y_1[ind]

                
                ind_1 = np.where(arr_1> thres_max_sum)
                _pass__ = False
                if len(ind_1) != 0:
                    arr_1 = arr_1[ind_1]
                    arr_2 = arr_2[ind_1]
                    arr_3 = arr_3[ind_1]
                    arr_4 = arr_4[ind_1]
                    arr_5 = arr_5[ind_1]
                    arr_6 = arr_6[ind_1]
                    arr_8 = arr_8[ind_1]
                    arr_y_1 = arr_y_1[ind_1]

                     # remove zeros
                    ind_7_0 = np.where(arr_y_1 != 0)
                    arr_1 = arr_1[ind_7_0]
                    arr_2 = arr_2[ind_7_0]
                    arr_3 = arr_3[ind_7_0]
                    arr_4 = arr_4[ind_7_0]
                    arr_5 = arr_5[ind_7_0]
                    arr_6 = arr_6[ind_7_0]
                    arr_8 = arr_8[ind_7_0]
                    arr_y_1 = arr_y_1[ind_7_0]

                    if len(arr_y_1) != 0:
                        # max of criterion, v/(sum* (mean**2)
                        ind_7 = np.where(arr_y_1==np.max(arr_y_1))
                        arr_1_1 = arr_1[ind_7]
                        arr_2_1 = arr_2[ind_7]
                        arr_3_1 = arr_3[ind_7]
                        arr_4_1 = arr_4[ind_7]
                        arr_5_1 = arr_5[ind_7]
                        arr_6_1 = arr_6[ind_7]
                        arr_8_1 = arr_8[ind_7] 
                        arr_y_1_1 = arr_y_1[ind_7]


                        # min of criterion, v/(sum* (mean**2)
                        ind_7 = np.where(arr_y_1==np.min(arr_y_1))
                        arr_1_0 = arr_1[ind_7]
                        arr_2_0 = arr_2[ind_7]
                        arr_3_0 = arr_3[ind_7]
                        arr_4_0 = arr_4[ind_7]
                        arr_5_0 = arr_5[ind_7]
                        arr_6_0 = arr_6[ind_7]
                        arr_8_0 = arr_8[ind_7] 
                        arr_y_1_0 = arr_y_1[ind_7]
                        
                        # max
                        inc.d_map  = arr_2_1[0]
                        inc.d_map_to_clusters()
                        inc.create_label()
                        rec.append(inc._labels)
                        var_original_map.append(np.var(arr_3_1[0]))
                        var_original_map_uni_.append(np.var(np.unique(arr_3_1[0],return_counts=True)[1]))
                        rec_arr_6.append(arr_6_1[0])
                        rec_arr_4.append(arr_4_1[0])
                        rec_arr_5.append(arr_5_1[0])
                        rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))
                        

                        # min
                        inc.d_map  = arr_2_0[0]
                        inc.d_map_to_clusters()
                        inc.create_label()
                        rec.append(inc._labels)
                        var_original_map.append(np.var(arr_3_0[0]))
                        var_original_map_uni_.append(np.var(np.unique(arr_3_0[0],return_counts=True)[1]))
                        rec_arr_6.append(arr_6_0[0])
                        rec_arr_4.append(arr_4_0[0])
                        rec_arr_5.append(arr_5_0[0])
                        rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))                        
                     
                    else:
                        # plan A
                        # arr_4 -> threshold(max clustering._record_dis_diff_sum)
                        original_labels = np.array(inc._original__.copy()) #(contain original self.d_map)  
                        arr_1 = inc.recor__.copy()  #(contain thresholds > 0, decides all the order of calculated results)
                        arr_2 = inc.labels_record__.copy() #(contain transformed self.d_map)
                        arr_3 = original_labels.copy()
                        arr_4 = np.array(inc._record_dis_diff_sum.copy()) # (based on original self.d_map)
                        arr_5 = np.array(inc._record_dis_diff_mean.copy()) # (based on original self.d_map)
                        arr_6 = np.array(inc._record_dis_diff_m_v.copy()) # (based on original self.d_map)
                        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            #             arr_7 = inc.clusters_record # (contain self.clusters)

                        n_clusters__ = []
                        for i in inc.clusters_record:
                            n_clusters__.append(len(i))
                        arr_8 = np.array(n_clusters__)


                        # basic condition !!!
                        ind = np.where(np.array(inc._record_dis_diff_sum) != None)
                        arr_1 = arr_1[ind]
                        arr_2 = arr_2[ind]
                        arr_3 = arr_3[ind]
                        arr_4 = arr_4[ind]
                        arr_5 = arr_5[ind]
                        arr_6 = arr_6[ind]
            #             arr_7 = arr_7[ind]
                        arr_8 = arr_8[ind]
                        arr_y_1 = (arr_6/arr_5)/arr_4
                        thres_max_sum = arr_1[np.where(arr_5==np.max(arr_5))][0]
                        _pass__ = True
                    
                elif len(ind_1) == 0:
                    # select threholds <= threshold corresponding to max sum
                    ind_1 = np.where(arr_1<= thres_max_sum)
                    arr_1 = arr_1[ind_1]
                    arr_2 = arr_2[ind_1]
                    arr_3 = arr_3[ind_1]
                    arr_4 = arr_4[ind_1]
                    arr_5 = arr_5[ind_1]
                    arr_6 = arr_6[ind_1]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    arr_8 = arr_8[ind_1]
                    arr_y_1 = arr_y_1[ind_1]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                     # remove zeros
                    ind_7_0 = np.where(arr_y_1 != 0)
                    arr_1 = arr_1[ind_7_0]
                    arr_2 = arr_2[ind_7_0]
                    arr_3 = arr_3[ind_7_0]
                    arr_4 = arr_4[ind_7_0]
                    arr_5 = arr_5[ind_7_0]
                    arr_6 = arr_6[ind_7_0]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    arr_8 = arr_8[ind_7_0]
                    arr_y_1 = arr_y_1[ind_7_0]


                    # max v/(sum* (mean**2)
                    ind_7 = np.where(arr_y_1==np.max(arr_y_1))
                    arr_1_1 = arr_1[ind_7]
                    arr_2_1 = arr_2[ind_7]
                    arr_3_1 = arr_3[ind_7]
                    arr_4_1 = arr_4[ind_7]
                    arr_5_1 = arr_5[ind_7]
                    arr_6_1 = arr_6[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #             rec_arr_7_f = []
        #             for i in ind_7[0]:
        #                 rec_arr_7_f.append(rec_arr_7[i])
                    arr_8_1 = arr_8[ind_7] 
                    arr_y_1_1 = arr_y_1[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                    # min v/(sum* (mean**2)
                    ind_7 = np.where(arr_y_1==np.min(arr_y_1))
                    arr_1_0 = arr_1[ind_7]
                    arr_2_0 = arr_2[ind_7]
                    arr_3_0 = arr_3[ind_7]
                    arr_4_0 = arr_4[ind_7]
                    arr_5_0 = arr_5[ind_7]
                    arr_6_0 = arr_6[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #             rec_arr_7_f = []
        #             for i in ind_7[0]:
        #                 rec_arr_7_f.append(rec_arr_7[i])
                    arr_8_0 = arr_8[ind_7] 
                    arr_y_1_0 = arr_y_1[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                    # max
                    inc.d_map  = arr_2_1[0]
                    inc.d_map_to_clusters()
                    inc.create_label()
                    rec.append(inc._labels)
                    var_original_map.append(np.var(arr_3_1[0]))
                    var_original_map_uni_.append(np.var(np.unique(arr_3_1[0],return_counts=True)[1]))
                    rec_arr_6.append(arr_6_1[0])
                    rec_arr_4.append(arr_4_1[0])
                    rec_arr_5.append(arr_5_1[0])
                    rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))


                    # min
                    inc.d_map  = arr_2_0[0]
                    inc.d_map_to_clusters()
                    inc.create_label()
                    rec.append(inc._labels)
                    var_original_map.append(np.var(arr_3_0[0]))
                    var_original_map_uni_.append(np.var(np.unique(arr_3_0[0],return_counts=True)[1]))
                    rec_arr_6.append(arr_6_0[0])
                    rec_arr_4.append(arr_4_0[0])
                    rec_arr_5.append(arr_5_0[0])
                    rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))
            
                if _pass__ == True:
                    _pass__ = False
                    # select threholds <= threshold corresponding to max sum
                    ind_1 = np.where(arr_1<= thres_max_sum)
                    arr_1 = arr_1[ind_1]
                    arr_2 = arr_2[ind_1]
                    arr_3 = arr_3[ind_1]
                    arr_4 = arr_4[ind_1]
                    arr_5 = arr_5[ind_1]
                    arr_6 = arr_6[ind_1]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    arr_8 = arr_8[ind_1]
                    arr_y_1 = arr_y_1[ind_1]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                     # remove zeros
                    ind_7_0 = np.where(arr_y_1 != 0)
                    arr_1 = arr_1[ind_7_0]
                    arr_2 = arr_2[ind_7_0]
                    arr_3 = arr_3[ind_7_0]
                    arr_4 = arr_4[ind_7_0]
                    arr_5 = arr_5[ind_7_0]
                    arr_6 = arr_6[ind_7_0]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    arr_8 = arr_8[ind_7_0]
                    arr_y_1 = arr_y_1[ind_7_0]


                    # max v/(sum* (mean**2)
                    ind_7 = np.where(arr_y_1==np.max(arr_y_1))
                    arr_1_1 = arr_1[ind_7]
                    arr_2_1 = arr_2[ind_7]
                    arr_3_1 = arr_3[ind_7]
                    arr_4_1 = arr_4[ind_7]
                    arr_5_1 = arr_5[ind_7]
                    arr_6_1 = arr_6[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #             rec_arr_7_f = []
        #             for i in ind_7[0]:
        #                 rec_arr_7_f.append(rec_arr_7[i])
                    arr_8_1 = arr_8[ind_7] 
                    arr_y_1_1 = arr_y_1[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


                    # min v/(sum* (mean**2)
                    ind_7 = np.where(arr_y_1==np.min(arr_y_1))
                    arr_1_0 = arr_1[ind_7]
                    arr_2_0 = arr_2[ind_7]
                    arr_3_0 = arr_3[ind_7]
                    arr_4_0 = arr_4[ind_7]
                    arr_5_0 = arr_5[ind_7]
                    arr_6_0 = arr_6[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #             rec_arr_7_f = []
        #             for i in ind_7[0]:
        #                 rec_arr_7_f.append(rec_arr_7[i])
                    arr_8_0 = arr_8[ind_7] 
                    arr_y_1_0 = arr_y_1[ind_7]
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



                    # max
                    inc.d_map  = arr_2_1[0]
                    inc.d_map_to_clusters()
                    inc.create_label()
                    rec.append(inc._labels)
                    var_original_map.append(np.var(arr_3_1[0]))
                    var_original_map_uni_.append(np.var(np.unique(arr_3_1[0],return_counts=True)[1]))
                    rec_arr_6.append(arr_6_1[0])
                    rec_arr_4.append(arr_4_1[0])
                    rec_arr_5.append(arr_5_1[0])
                    rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))
                    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    

                    # min
                    inc.d_map  = arr_2_0[0]
                    inc.d_map_to_clusters()
                    inc.create_label()
                    rec.append(inc._labels)
                    var_original_map.append(np.var(arr_3_0[0]))
                    var_original_map_uni_.append(np.var(np.unique(arr_3_0[0],return_counts=True)[1]))
                    rec_arr_6.append(arr_6_0[0])
                    rec_arr_4.append(arr_4_0[0])
                    rec_arr_5.append(arr_5_0[0])
                    rec_distances_to_centers.append(distances_to_centers(inc._labels, inc._data_))
        #------------------------------------------------encoding process----------------------------------------------
        # step 2: decoding process
        # start second selection
        inc.second_selection()
        
    def second_selection(self):
        """determine final labels"""
        a = np.array(var_original_map)
        b= np.array(var_original_map_uni_)
        c = np.array(rec_arr_6)
        d = np.array(rec_arr_4)  # sum
        e = np.array(rec_arr_5) # mean
        f = np.array(rec_distances_to_centers)
        g = np.array(rec)

        c_1 = rearrange(b, c)
        d_1 = rearrange(b, d)
        e_1 = rearrange(b, e)
        f_1 = rearrange(b, f)
        g_1 = rearrange(b, g)


        b_1 = np.array(sorted(var_original_map_uni_, reverse= False))
        b_1, ind__ = np.unique(b_1, return_index = True)

        c_1 = c_1[ind__]
        e_1 = e_1[ind__]
        d_1 = d_1[ind__]
        f_1 = f_1[ind__]
        g_1 = g_1[ind__]

        _res = []
        for i in range(0, len(c_1)-10):
            _res.append(np.var(c_1[i:i+10]))


        # determine (decline, stop) area 
        d_ = _res
        diff_ = [(lambda a, b  : a-b)(d_ [i], d_ [i-1]) for i in range(1,len(d_ ))]
        diff_ = np.array(diff_)


        diff_ = np.insert(diff_, 0 , 0)
        partition_1 = []  # record decline start
        partition_2 = []  # record increase start
        partition_3 = []  # record stop point

        flag_1 = ''
        flag_2 = 'record increase start'
        flag_3 = ''

        for i in range(0, len(diff_)):
            decline_detection_ = diff_[i:i+2]
            increase_detection = diff_[i:i+2]
            stop_detection = diff_[i:i+2]

            # record increase start
            if flag_2 == 'record increase start' and len(np.where(increase_detection>0)[0]) == 2:
                partition_2.append(i)
                flag_1 = 'record decline start'
                flag_2 = 'do not record increase start'


            # only after increase, record decline start
            if flag_1 == 'record decline start' and len(np.where(decline_detection_<0)[0]) == 2:
                partition_1.append(i)   
                flag_1 = 'do not record decline start'
                flag_2 = 'record increase start'


            if len(partition_1) ==1 and len(partition_2) == 2:
                flag_3 = 'ready to stop'

            if  flag_3 == 'ready to stop' and len(np.where(stop_detection>0)[0]) == 1 and len(np.where(stop_detection>0)[0]) == 1:
                partition_3.append(i+1)
                break


        # if decline and stop exist, select minimum of dictances to center within area and corresponding d':
        if  len(partition_3)!= 0: 
            start_ = partition_1[0]
            end_ = partition_3[0]

            F_local_ = np.array(f_1[start_:end_+1])
            base_ = np.where(f_1 == np.min(F_local_))[0]
            first_ = base_[np.where(base_ >= start_)]
            second_ = first_[np.where(first_ <= end_ )]
            final_labels = g_1[second_[0]]

            # final clustering labels
            self.f__labels__ = final_labels.copy()


        # if decline and stop don't exist, select most repeated minimum of dictances to center:    
        elif len(partition_3)== 0: 
            base_s_ = np.unique(f, return_counts=True )
            min_re_ = base_s_[0][np.where(base_s_[1] == np.max(base_s_[1]))][0]
            index_final_ = np.where(f == min_re_)[0][0]
            final_labels = g[index_final_]
            # final clustering labels
            self.f__labels__ = final_labels.copy()   
            
def transform_2(arg):
    '''reshape from (a, b) to (a, 1, b)
    parameters:
    -----------------------
    arg: array with shape (a, b)

    return:
    array with shape (a, 1, b)
    '''
    if len(arg.shape) != 2:
        raise ValueError("arg shape must be (a, b)")
    h, d = arg.shape
    w = 1
    return np.reshape(arg, (h, w, d))

            
def rearrange_col(arg_1, arg_2, ord_ = 'inc'):
    '''
    sort arg_2 along columns

    parameter:
    --------------
    arg_1 - row vector to sort (step 1), array, shape (a,)
    arg_2 - matrix to rearrange according to step 1, array, shape (a, b)
    ord_ - order of arg_2, 'inc' or 'dec', string,
            'inc':increase, for exmaple, 1,2,3,4
            'dec':decrease, for exmaple, 4,3,2,1
    return
    -------------
    result - sorted arg_2 according to ord_, array'''

    order = np.argsort(arg_1, axis = None)
    def _reverse_(arg):
        '''reverse arg
        parameter:
        ----------
        arg: data to be reversed, array

        return:
        ----------
        reversed arg, array
        '''
        _length_ = len(arg)
        return np.array([arg[_length_-1-i] for i in range(0,_length_)])
    if ord_ == 'inc': # increase, for exmaple, 1,2,3,4
        arg_2_rearranged = [arg_2[:,i] for i in order]
        result = np.array(arg_2_rearranged).transpose()
    elif ord_ == 'dec': # decrease, for exmaple, 4,3,2,1
        order = _reverse_(order)
        arg_2_rearranged = [arg_2[:,i] for i in order]
        result = np.array(arg_2_rearranged).transpose()   
    return result
    

def position_index(arg):
    if isinstance(arg, (np.ndarray)) == False:
        raise ValueError("input must be array")
    h, w, d = arg.shape
    array_1 = np.reshape(arg, (h*w, d))
    position = [[] for i in range(0,h)]
    for i in range(0,h):
        position[i]=[[] for j in range(0,w) ]
    for x in range(0,h):
        for y in range(0,w):
            position[x][y].append(x)
            position[x][y].append(y)
    position_1 = np.array(position)
    h_1, w_1, d_1 = position_1.shape
    position_index = np.reshape(position_1, (h_1*w_1, d_1))
    return np.hstack((position_index,array_1))

def rearrange(arg_1, arg_2):
    if isinstance(arg_1, (np.ndarray)) == False:
        raise ValueError("input must be array")
    if isinstance(arg_2, (np.ndarray)) == False:
        raise ValueError("input must be array")
    order = np.argsort(arg_1, axis = None)
    arg_2_rearranged = [arg_2[i] for i in order]
    return np.array(arg_2_rearranged)


def distances_to_centers(_labels_, _data_):
    '''parameters:
    -----------------
    _labels: obtained clusters' labels for _data, array, shape (a, )
    _data: data, array, shape (a, b)
    
    returns:
    -----------------
    result: total sum of distances from datapoints to clusters' centers , numpy.int32
    
    '''
    _labels = _labels_.copy()
    _data = _data_.copy()
    _distances = []
    for _label in set(_labels):
        data_vec = _data[np.where(_labels ==_label)].copy()
        center_vec = np.mean(data_vec, axis=0).copy()
        _distances.append(np.linalg.norm(center_vec - data_vec))
    _distances = np.array(_distances)                     
    result = np.sum(_distances)
    return result


from sklearn.ensemble import RandomForestClassifier
def get_importances(data, data_instance, sort_column, s_c, seed_s):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # sort column
    if sort_column == True:
        _to_s_ = data[:,s_c].copy()
        data = rearrange(_to_s_ , data)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    forest = RandomForestClassifier(random_state=0)
    
    # choose data and target and rerrange
    def _choose_index_(_data):
# #         rng = np.random.default_rng(seed = 1)
#         _index = rng.integers(0,len(_data)-1, size= (round(len(_data)*0.9),))
        rng = np.random.default_rng(seed = seed_s)
        _index = rng.choice(len(_data)-1, size= (round(len(_data)*0.7),) , replace=False)
        return _index

    r_d = []
    r_t = []
    for i in set(data_instance.target):
        d_1 = data[np.where(data_instance.target==i)]
        t_1 = data_instance.target[np.where(data_instance.target==i)]
        c_i = _choose_index_(d_1)
        r_d.append(d_1[c_i])
        r_t.append(t_1[c_i])
    r_d_c = r_d[0]
    r_t_c = r_t[0]
    for ind in range(0, len(r_d)-1):
        # rerranged data
        r_d_c = np.vstack((r_d_c,r_d[ind+1]))
        # rerranged target
        r_t_c = np.append(r_t_c,r_t[ind+1])
    #---------------------------------------------------------------
    # calculate importances
    forest.fit(r_d_c, r_t_c)
    importances = forest.feature_importances_
    print('importances')
    print(importances)
    return importances

