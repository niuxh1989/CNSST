# CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering


import numpy as np
from PIL import Image

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

class CNSST:
    def __init__(self, X, adjuster = 90, sensitivity = 0, data_instance = None, n_p = 10, n_clusters = None, original_constants = np.array([1, 5, 10]).astype(float), weights_ = np.array([1, 1.05, 1.1]).astype(float), option_for_order_of_target = '' ):

        """Perform CNSST clustering and edge detection from input array X

        CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering

        Parameters
        -----------
        X: array, look at example, shape can be (a, b, c), (a, b)
        adjuster: default = 90
        sensitivity:  default = 0

        Returns
        -------
         _labels : ndarray
         Cluster labels for each point

        """

        self.adjuster = adjuster
        self.sensitivity = sensitivity
        self.o_c = 0
        self.weights__ = 0
        self.o_c_stor_ = original_constants
        self.weights__stor_ = weights_

        self.o_c_stor_init__ = np.array([1, 5, 10]).astype(float)
        self.weights_init__ = np.array([1, 1.05, 1.1]).astype(float)

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
        self.array_1 = (X/adjuster).astype(float).copy()
        self.array = position_index(self.array_1).copy()
        self.adjuster = adjuster

        self.e_detected = np.zeros((self.y_axis, self.x_axis, 3),dtype=np.uint8)

        self.data_instance = data_instance
        self.target = data_instance.target
        self.n_clusters = n_clusters
        # initiate
        self.n_ = n_p
        # features' influence
        self.impact_record = []
        # target order
        self.order ='dir_1'
        self.option = option_for_order_of_target #'no order': there is no order in target
        #
        self.multi_ = [ 2, 1.9, 1.8, 1.7, 1.5, 1, 0.5, 0.2, 0.01, 0.001]
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

    def __F_function(self, array_):
        a_3 = array_
        h, w = a_3.shape
        d = w -2
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
            alpha_vector[:,0] = generate_constants(d, arg)
            return alpha_vector


        def initial_weights(d, arg):
            '''initial_weights
            parameters:
            ------------------
            d: dimension
            arg: weights for dimension of three np.array([1,1,2])

            return:
            ------------------
            an array wirth shape(d, 1)'''
            result = generate_constants(d, arg)
            print('weights')                                 # -------------------------------------------------------------------------
            print(result)
            return result

        def initial_matrix(d, arg):
            '''create a zero matrix for genrating matrix for all single points
                ---------------------
                d: dimension'''
            _matrix = np.zeros((d, d),dtype = float)
            _matrix[0:(d+1),:] = arg
            return _matrix


        def _fifth_column(arg):

            '''
            parameters:
            ---------------------
            arg: matrix

            return:
            ---------------------
            add a fifth column for matrix
            '''

            A = arg[1,:]
            A = np.insert(A, 0, A[len(A)-1])
            i = len(A)-1
            A = np.delete(A, i,axis = None)
            return A


        def transform_1(arg):
            if len(arg.shape) != 1:
                raise ValueError("arg shape must be (a, )")
            h = arg.shape[0]
            w = 1
            d = 1
            return np.reshape(arg, (h*w, d))

        def un__F_function(a_3, original_constants = self.o_c, weights_ = self.weights__):
            '''calculate F function
            ------------------------
            a_3: data matrix
            weights_ = np.array([1, 1.15, 1.2])
            original_constants (previous): np.array([1, 1.22, 2.44]).astype(float)


            original_constants: np.array([1, 1.22, 1.28])'''

            # print(locals())
            def f_s(d):
                '''create a zero matrix for filling with fs'''
                return np.zeros((d, 1),dtype=float)
            vector_1 = initial_alpha(d, original_constants).copy()
            # print('vector_1')
            # print(vector_1)

            e = np.ones(((d, d)),dtype=float)

            f_ = f_s(d)
            weights = initial_weights(d, weights_) # generation of weights
            f_2_ = []
            for i in range(0, len(a_3)):
                matrix_1 = (vector_1*initial_matrix(d, a_3[i][2:2+d]) + e)**2
                matrix_2 = _fifth_column(matrix_1)
                const_ = np.sum(matrix_1[0, 1:d])
                for n in range(0,len(f_)):

                    sq_sum = np.sum(matrix_1[n, 0]) + const_

                    two_sq = matrix_1[0, n]
                    _sq = matrix_2[n]

                    f_[n] = np.sqrt(((sq_sum-two_sq)/sq_sum)*(two_sq/(sq_sum-_sq)))
                f_1_ = np.sum((f_*weights)**3)
                f_2_.append(f_1_)
            return np.array(f_2_)
        #
        #
        self.F = un__F_function(array_)

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
                self.threshold_index = round(i+((len(self.Y_increment_function_2)-i)*self.sensitivity/100))
                break

    def run(self):
        # self.set__par_un()
        self.__F_function(self.array)
        self.sorting()
        self.diff()
        self.thresh_()
        self.cluster()
        self.create_label()
#         self.fast_plot()
        return self


    def fast_plot(self):
        self.image = np.zeros((self.y_axis, self.x_axis, 3),dtype=np.uint8)
        self.array_2 = position_index(self.X.astype(int)).copy()
        _x = rearrange(self.F, self.array[:,0]).astype(int) # row    #------------------------------------------------------------------
        _y = rearrange(self.F, self.array[:,1]).astype(int) # column #------------------------------------------------------------------
        R = rearrange(self.F, self.array_2[:,2])
        G = rearrange(self.F, self.array_2[:,3])
        B = rearrange(self.F, self.array_2[:,4])
        for m in range(0,len(self.clusters)):
            r = R[self.clusters[m]].mean()
            g = G[self.clusters[m]].mean()
            b = B[self.clusters[m]].mean()
            for n in range(0,len(self.clusters[m])):
                self.image[_x[self.clusters[m][n]],_y[self.clusters[m][n]]] = [r, g, b]
        self.im = Image.fromarray(self.image)
        self.im.show()


    def save_file_c_c(self, path):
        self.im.save(path)


    def save_file_e_(self, path):
        self.ed.save(path)


class _edge_(CNSST): #(improving)--------------------------------------------------------------------------------
    def edge(self):
        for m in range(0, self.y_axis):
            self.r_or_col = self.array[np.where(self.array[:,0]== m)].copy()
            self.run_edge()
            self.edge_horizontal_direction = self.edge_cluster
            # plot on empty page
            self.plot_edge(self.F, self.edge_horizontal_direction, self.r_or_col)

        for n in range(0, self.x_axis):
            self.r_or_col = self.array[np.where(self.array[:,1]== n )].copy()
            self.run_edge()
            self.edge_vertical_direction = self.edge_cluster
            self.plot_edge(self.F, self.edge_vertical_direction, self.r_or_col)
        self.ed = Image.fromarray(self.e_detected)
        self.ed.show()
        return self

    # def set__par_ed(self):
    #     self.o_c = self.o_c_stor_
    #     self.weights__ = self.weights__stor_

    def run_edge(self):
        # self.set__par_ed()
        self.__F_function(self.r_or_col)
        self.sorting()
        self.diff_edge()
        self.thresh_()
        self.cluster_edge()
        self.get_edge_cluster()
        return self

    def diff_edge(self):
        """Signal contrast for edge detection"""
        self.Y_increment_function_1 = [(lambda a, b  : abs(a-b))(self.F_sorted[i], self.F_sorted[i-1]) for i in range(1,len(self.F_sorted))]
        # add 0 in front of list, because lenghth of Y_increment_function_1 is less than F_sorted by one
        self.Y_increment_function_1.insert(0,0)

    def cluster_edge(self):
        self.threshold = float(self.Y_increment_function_2[self.threshold_index])
        self.clusters=[[0]]
        for i in range(1, (len(self.Y_increment_function_1)-1)):
            if (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) > 0:
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) < 0:
                self.clusters.append([])
                self.clusters[len(self.clusters)-1].append(i)
            elif (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i]-self.threshold) == 0 and (self.Y_increment_function_1[i-1]-self.threshold)*(self.Y_increment_function_1[i+1]-self.threshold) > 0:
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

    def get_edge_cluster(self):
        self.edge_cluster=[]
        for m_e in range(0, len(self.clusters)):
            if self.Y_increment_function_1[self.clusters[m_e][0]] >= self.threshold:
                self.edge_cluster.append(self.clusters[m_e])


    def plot_edge(self, F, edge_cluster, data):
        _x_ = rearrange(F, data[:,0]).copy() # row
        _y_ = rearrange(F, data[:,1]).copy() # column
        for m in range(0,len(edge_cluster)):
            for n in range(0,len(edge_cluster[m])):
                self.e_detected[_x_[edge_cluster[m][n]],_y_[edge_cluster[m][n]]] = [255, 255, 255]
        return

class supervised_CNSST(CNSST):
    @staticmethod
    def predict(inc, data, target):
        inc.data_test = data
        if len(inc.data_test.shape) == 2:
            inc.data_test = transform_2(inc.data_test)

        y_axis, x_axis, dim = inc.data_test.shape
        inc.x_axis = x_axis
        inc.y_axis = y_axis
        inc.x_start = 0
        inc.x_end = x_axis
        inc.y_start = 0
        inc.y_end = y_axis

        inc.target_test = target
        inc.set__par_1()
        inc.array_test = position_index((inc.data_test/inc.adjuster).astype(float)).copy()
        h, w = inc.array_test.shape
        w = w-2
        for i in range(0, w):
            inc.array_test[:,i+2] = inc.array_test[:,i+2]*(inc.dic_for_test[i]*inc.dic_for_test[i])
        inc._CNSST__F_function(inc.array_test)
        inc.sorting()
        inc.diff_supervised()
        inc.cluster_supervised()
        inc.create_label_sup()
        copy = inc._labels.copy()
        inc._labels_test = inc.transform_labels___(copy, n_clusters = inc.n_clusters )
        return inc

    def prepare_(self):
        self.set__par_t()
        self.run()
        self.thresh_supervised(self.n_) # set initial threshold
        self.significance_f()
        return self

    def set__par_t(self):
        self.o_c = self.o_c_stor_init__
        self.weights__ = self.weights_init__

    def set__par_1(self):
        self.o_c = self.o_c_stor_
        self.weights__ = self.weights__stor_

    def run_supervised_1(self):
        # self.run()
        self.set__par_1()
        self._CNSST__F_function(self.array)
        self.sorting()
        self.diff()
        return self.adjust_sensitivity()

    def significance_f(self):
        '''
            record how well features can characterize result

        '''

        # change self.array using number 10 and without 10
        h, w = self.array.shape
        w = w -2

        self.no_mutiplier = self.array.copy()
        self._im_record_total_ = []

        for multi_ in self.multi_:
            for i in range(0, w):
                self.array[:,2:(len(self.array[0])-1)] = self.array[:,2:(len(self.array[0])-1)]*multi_
                self.array[:,i+2] = self.array[:,i+2]*multi_
                check_len_1 = len(self.impact_record)
                self.test_prep()
                self.check(self.data_instance.target, self._labels)
                check_len_2 = len(self.impact_record)
                if self.order == 'dir_1':
                    pass
                else:
                    self.change_target(self.data_instance.target)
                    self.test_prep()
                    if self.flag_3__ == 'n' and check_len_2 == check_len_1:
                        self._labels = self.transform_labels___(self._labels.copy(), self.n_clusters)
                        objective__1_ = self._similarity_(self._labels, self.target)
                        self.impact_record.append(objective__1_)

                # reset
                self.array[:,2:(len(self.array[0])-1)] = self.array[:,2:(len(self.array[0])-1)]/multi_
                self.array[:,i+2] = self.array[:,i+2]/multi_
                self.target  = self.data_instance.target
                self.flag_3__ = 'no_flag'
            self._im_record_total_.append(self.impact_record)
            self.impact_record = []

        _im_record_= np.array(self._im_record_total_)
        index_rec__ = np.zeros((w,), dtype=float)
        for i in range(0, w):
            index_rec__[i]= np.where(_im_record_[:,i]==np.min(_im_record_[:,i]))[0][0]
        to_fac__= np.array(self.multi_)
        ex_in_to_fac_ = dict(zip(np.array([i for i in range(0, len(to_fac__))]), to_fac__))
        for ind in range(0,len(index_rec__)):
            index_rec__[ind] = ex_in_to_fac_[index_rec__[ind]]
        self.impact_record = index_rec__
        _dict__ = dict(zip(np.array([i for i in range(0,w)]), self.impact_record))
        self.dic_for_test = _dict__
        for i in range(0, w):
            self.array[:,i+2] = self.array[:,i+2]*(_dict__[i]*_dict__[i])
        self.set__par_1()
        self.run()
        self.check(self.data_instance.target, self._labels)
        self.change_target(self.data_instance.target)

    def test_prep(self):
        self.fit_for_test()
        return self

    def fit_for_test(self):
        rate = 0.01*self.threshold
        self._labels = self.transform_labels___(self._labels.copy(), self.n_clusters)
        objective_1 = self._similarity_(self._labels, self.target)
        flag_1 = objective_1
        flag_2 = 'pos'
        # flag_3 = 'y'
        def adjust_threshold(threshold, flag, rate):
            if flag == 'pos':
                threshold += rate
            else:
                threshold -= rate
            return threshold
        count_rate = [10,100]
        while abs(count_rate[len(count_rate)-1]-count_rate[len(count_rate)-2]) > 0.000000001:
            self.threshold = adjust_threshold(self.threshold, flag_2, rate)
            self.run_supervised_2()
            if self.option == 'no order':
                 pass
            elif self.option != 'no order':
                con_1 = (np.mean(np.where(self.target==np.max(self.target))) - np.mean(np.where(self.target==np.min(self.target))))
                con_2 = (np.mean(np.where(self._labels==np.max(self._labels))) - np.mean(np.where(self._labels==np.min(self._labels))))
                if con_1*con_2 < 0:
                    flag_3 = 'n'
                    self.flag_3__  = flag_3
                    break
                elif con_1*con_2 >= 0:
                    pass
            flag_3 = 'y'
            self.flag_3__ = flag_3
            self._labels = self.transform_labels___(self._labels.copy(), self.n_clusters)
            objective_ = self._similarity_(self._labels, self.target)
            if flag_2 == 'pos' and objective_ > flag_1:
                flag_2 = 'neg'
                rate = rate/10
            elif flag_2 == 'neg' and objective_ > flag_1:
                flag_2 = 'pos'
                rate = rate/10
            flag_1 = objective_
            count_rate.append(self.threshold)
        if flag_3 == 'y':
            self.impact_record.append(flag_1)
        return self


    def check(self, arg_1, arg_2):
        '''
        >0 : i
        <0 : d
        =0 : not d and not i

        remind:
        -------------------------------
        order: 'dir_1' (no change)
               'dir_2' (des order)
               'dir_3' (inc order)'''
        if self.option == 'no order':
            self.order = 'dir_1'
        elif self.option != 'no order':
            if (np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))))*(np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2)))) >= 0:
                self.order = 'dir_1'
            elif (np.mean(np.where(arg_1==np.max(arg_1))) - np.mean(np.where(arg_1==np.min(arg_1))))*(np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2)))) < 0:
                if np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2))) < 0:
                    self.order = 'dir_2'
                elif np.mean(np.where(arg_2==np.max(arg_2))) - np.mean(np.where(arg_2==np.min(arg_2))) > 0:
                    self.order = 'dir_3'
        return

    def adjust_sensitivity(self):
        return  self.fit_threshold()

    def fit_threshold(self):
        '''
        run clustering, get self._labels
        objective_1
        change threshold
        run clustering, get self._labels
        objective_2'''
        self.objective_record = []
        self.labels_record__ = []
        for thres_ in self.recor__:
            self.threshold = thres_
            # print('self.threshold')
            # print(self.threshold)
            # self.run_supervised_2()
            try:
                self.run_supervised_2()
            except ValueError:
                pass
            self._labels = self.transform_labels___(self._labels.copy(), self.n_clusters)
            objective_ = self._similarity_(self._labels, self.target)
            self.objective_record.append(objective_)
            self.labels_record__.append(self._labels)
        self.objective_record = np.array(self.objective_record)
        self.labels_record__ = np.array(self.labels_record__)

        self.threshold = self.recor__[np.where(self.objective_record==np.min(self.objective_record))]
        self._labels = self.labels_record__[np.where(self.objective_record==np.min(self.objective_record))]
        if len(self._labels.shape)!= 1:
           record_dis_diff_sum_r = []
           for _labels__in in self._labels:
              re_arr_, re_arr_counts_ = np.unique(_labels__in, return_counts=True)
              # mean value of positions of all clusters
              record_dis = []
              for i in range(0,len(set(_labels__in))):
                  record_dis.append(np.mean(np.where(_labels__in==i)))
              # absolute value of distance between clusters
              record_dis_diff_sum_r.append(np.sum(np.array([(lambda a, b  : abs(a-b))(record_dis[i], record_dis[i-1]) for i in range(1,len(record_dis))])))
           record_dis_diff_sum_r = np.array(record_dis_diff_sum_r)
           self._labels = self._labels[np.where(record_dis_diff_sum_r==np.max(record_dis_diff_sum_r))]
           self.threshold = self.threshold[np.where(record_dis_diff_sum_r==np.max(record_dis_diff_sum_r))]
           if len(self._labels.shape)!= 1:
              self._labels = self._labels[np.where(self.threshold==np.max(self.threshold))]
              h, w = self._labels.shape
              self._labels = np.reshape(self._labels, (w,))
              self.threshold = np.max(self.threshold)
        self._labels = self.transform_labels___(self._labels.copy(), self.n_clusters)
        objective_f = self._similarity_(self._labels, self.target)
        return self


    def run_supervised_2(self):
        self._CNSST__F_function(self.array)
        self.sorting()
        self.diff_supervised()
        self.cluster_supervised()
        self.create_label()
        return self

    def diff_supervised(self):
        self.Y_increment_function_1 = [(lambda a, b  : a-b)(self.F_sorted[i], self.F_sorted[i-1]) for i in range(1,len(self.F_sorted))]
        self.Y_increment_function_1.insert(0,0)
#         self.Y_increment_function_1 = list(np.round(np.array(self.Y_increment_function_1), decimals=3))


    def thresh_supervised(self, n_):
        self.hist = np.histogram(self.recor__, bins = len(set(self._labels)))
        self.threshold = (self.hist[1][np.where(self.hist[0]==np.max(self.hist[0]))]+self.hist[1][np.where(self.hist[0]==np.max(self.hist[0]))+np.array([1])])/2


    def cluster_supervised(self):
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
    def change_target(self, arg):  # for supervised training
        '''
        parameters:
        ------------------
        arg: array of target
        order: 'dir_1' (no change)
               'dir_2' (des order)
               'dir_3' (inc order)
        return:
        ------------------
        array with changed (or unchanged ) order of target
        '''
        if self.option == 'no order':
            self.order = 'dir_1'
        order = self.order
        if order == 'dir_1': # no change
            self.target = arg
        elif order == 'dir_2': # des
            fir_ = sorted(list(set(arg)), reverse=False)
            sec_ = sorted([i for i in range(0, len(fir_))] , reverse=True)
            ex_ = dict(zip(np.array(fir_), np.array(sec_ )))
            for ind in range(0,len(arg)):
                arg[ind] = ex_[arg[ind]]
            self.target = arg
        elif order == 'dir_3': # inc
            fir_ = sorted(list(set(arg)), reverse=True)
            sec_ = sorted([i for i in range(0, len(fir_))] , reverse=False)
            ex_ = dict(zip(np.array(fir_), np.array(sec_ )))
            for ind in range(0,len(arg)):
                arg[ind] = ex_[arg[ind]]
            self.target = arg

    def transform_labels___(self, _labels__, n_clusters):
        if len(set(_labels__)) == 1:
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
            print(merged_clusters)

            s_m = merged_clusters
            _target_w_ = [i for i in range(0, len(set(_labels__)))]
            _target_x_ = _target_w_.copy()
            for i in range(0, len(merged_clusters)):
                for m in range(0, len(s_m[i])):
                    _target_w_[s_m[i][m]] = i
            ex_dict__ = dict(zip(np.array(_target_x_), np.array(_target_w_)))
            print(ex_dict__)
            for ind in range(0,len(_labels__)):
                _labels__[ind] = ex_dict__[_labels__[ind]]
        return _labels__

    def _similarity_(self, __label__, __target__):
        _r1_, _r2_ = np.unique(__label__ - __target__ , return_counts=True)
        ex_dic_ = dict(zip(_r1_, _r2_))
        return 1/(ex_dic_[0]/np.sum(_r2_))

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
