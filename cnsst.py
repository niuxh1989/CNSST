#!/usr/bin/env python
# coding: utf-8
# author: Niu Xinghao, 
#         Ph.D. student
#         Department of Economic informatics
#         Faculty of Economics
#         Moscow State University

# In[ ]:


# Perform CNSST clustering and edge detection from input array X
# CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering

# How to use:
# # Example
# # --------------------
# from PIL import Image
# path = eval(input())
# test_image = Image.open(path)
# X = np.array(test_image)

# # for color clustering run:
# clustering = CNSST(X).run()
# clustering._labels
# clustering.save_file_c_c(path for saving result)
        
# # for edge detection run:
# edge_ = CNSST(X, adjuster = 5).edge()
# edge_.save_file_e_(path for saving result)

import numpy as np
from PIL import Image

class CNSST:
    def __init__(self, X, adjuster = 90, sensitivity = 0):
        
        """Perform CNSST clustering and edge detectionfrom input array X
        
        CNSST: Clustering algorithm based on Network signal transformation, Sorting, Signal contrast and Threshold filtering
       
        Parameters
        -----------
        X: array, look at example
        adjuster: default = 90
        sensitivity:  default = 0
        
        Returns
        -------
         _labels : ndarray
         Cluster labels for each point
        
        """
        self.adjuster = adjuster
        self.sensitivity = sensitivity
        y_axis, x_axis, dim = X.shape
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_start = 0
        self.x_end = x_axis
        self.y_start = 0
        self.y_end = y_axis
        self.array_1 = (X/adjuster).astype(int).copy()
        self.array = position_index(self.array_1).copy()
        
        self.e_detected = np.zeros((self.y_axis, self.x_axis, 3),dtype=np.uint8)
        
        
    def cluster(self):
        self.threshold = float(self.Y_increment_function_2[self.threshold_index])
        self.clusters=[[0]]
        for i in range(1, (len(self.Y_increment_function_1))):
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
    
    def cluster_edge(self):
        self.threshold = float(self.Y_increment_function_2[self.threshold_index])

        self.clusters=[[0]]
        for i in range(1, (len(self.Y_increment_function_1))):
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
    
    def create_label(self):
        self.labels = np.zeros((self.y_axis,self.x_axis))
        self.x = rearrange(self.F, self.array[:,0])
        self.y = rearrange(self.F, self.array[:,1])
        for m in range(0,len(self.clusters)):
            for n in range(0,len(self.clusters[m])):
                self.labels[self.x[self.clusters[m][n]],self.y[self.clusters[m][n]]] = m
        h, w = self.labels.shape
        self._labels = np.reshape(self.labels, (h*w))
        
    def __F_function(self, array_):
        self.f_1 = (np.sqrt((array_[:,3]+1)*(array_[:,3]+1)+(array_[:,4]+1)*(array_[:,4]+1))/np.sqrt((array_[:,2]+1)*(array_[:,2]+1)+(array_[:,3]+1)*(array_[:,3]+1)+(array_[:,4]+1)*(array_[:,4]+1)))*(np.sqrt((array_[:,2]+1)*(array_[:,2]+1))/np.sqrt((array_[:,2]+1)*(array_[:,2]+1)+(array_[:,3]+1)*(array_[:,3]+1)))
        self.f_2 = (np.sqrt((1.28*array_[:,2]+1)*(1.28*array_[:,2]+1)+(array_[:,4]+1)*(array_[:,4]+1))/np.sqrt((1.28*array_[:,2]+1)*(1.28*array_[:,2]+1)+(array_[:,3]+1)*(array_[:,3]+1)+(array_[:,4]+1)*(array_[:,4]+1)))*(np.sqrt((array_[:,3]+1)*(array_[:,3]+1))/np.sqrt((array_[:,3]+1)*(array_[:,3]+1)+(array_[:,4]+1)*(array_[:,4]+1)))*2
        self.f_3 = (np.sqrt((1.22*array_[:,2]+1)*(1.22*array_[:,2]+1)+(array_[:,3]+1)*(array_[:,3]+1))/np.sqrt((1.22*array_[:,2]+1)*(1.22*array_[:,2]+1)+(array_[:,3]+1)*(array_[:,3]+1)+(array_[:,4]+1)*(array_[:,4]+1)))*(np.sqrt((array_[:,4]+1)*(array_[:,4]+1))/np.sqrt((1.22*array_[:,2]+1)*(1.22*array_[:,2]+1)+(array_[:,4]+1)*(array_[:,4]+1)))
        self.F = self.f_1*self.f_1*self.f_1+self.f_2*self.f_2*self.f_2+self.f_3*self.f_3*self.f_3
        
    def diff(self):
        self.Y_increment_function_1 = [(lambda a, b  : a-b)(self.F_sorted[i], self.F_sorted[i-1]) for i in range(1,len(self.F_sorted))]
        # add 0 in front of list, because lenghth of Y_increment_function_1 is less than F_sorted by one
        self.Y_increment_function_1.insert(0,0)
        
    def diff_edge(self):
        self.Y_increment_function_1 = [(lambda a, b  : abs(a-b))(self.F_sorted[i], self.F_sorted[i-1]) for i in range(1,len(self.F_sorted))]
        # add 0 in front of list, because lenghth of Y_increment_function_1 is less than F_sorted by one
        self.Y_increment_function_1.insert(0,0)
        
    def edge(self):
        for m in range(0, self.y_axis): 
            self.r_or_col = self.array[np.where(self.array[:,0]== m)].copy()  # select m-th row
            self.run_edge()
            self.edge_horizontal_direction = self.edge_cluster # edge cluster of horizontal direction for every loop
            # plot on empty page
            self.plot_edge(self.F, self.edge_horizontal_direction, self.r_or_col)
            
            
        for n in range(0, self.x_axis): 
            self.r_or_col = self.array[np.where(self.array[:,1]== n )].copy() # select n-th column
            self.run_edge()
            self.edge_vertical_direction = self.edge_cluster # edge cluster of vertical direction for every loop
            # plot on empty page
            self.plot_edge(self.F, self.edge_vertical_direction, self.r_or_col)
        self.ed = Image.fromarray(self.e_detected)
        self.ed.show()  
        return self
    
    def plot_edge(self, F, edge_cluster, data):
        _x_ = rearrange(F, data[:,0]).copy() # row
        _y_ = rearrange(F, data[:,1]).copy() # column
        for m in range(0,len(edge_cluster)):
            for n in range(0,len(edge_cluster[m])):
                self.e_detected[_x_[edge_cluster[m][n]],_y_[edge_cluster[m][n]]] = [255, 255, 255] 
        return
    
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
        self.fast_plot()
        return self
    
    def run_edge(self): 
        self.__F_function(self.r_or_col)
        self.sorting()
        self.diff_edge()
        self.thresh_()
        self.cluster_edge()
        self.get_edge_cluster()
        return self
    

    def get_edge_cluster(self):
        self.edge_cluster=[]
        for m_e in range(0, len(self.clusters)):
            if self.Y_increment_function_1[self.clusters[m_e][0]] >= self.threshold:
                self.edge_cluster.append(self.clusters[m_e])

    
    def fast_plot(self):
        """a fast way for plotting result by using mean pixel value of cluster"""
        self.image = np.zeros((self.y_axis, self.x_axis, 3),dtype=np.uint8)
        _x = rearrange(self.F, self.array[:,0]) # row
        _y = rearrange(self.F, self.array[:,1]) # column
        R = rearrange(self.F, self.array[:,2])
        G = rearrange(self.F, self.array[:,3])
        B = rearrange(self.F, self.array[:,4])
        for m in range(0,len(self.clusters)):
            r = R[self.clusters[m]].mean()*self.adjuster
            g = G[self.clusters[m]].mean()*self.adjuster
            b = B[self.clusters[m]].mean()*self.adjuster
            for n in range(0,len(self.clusters[m])):
                self.image[_x[self.clusters[m][n]],_y[self.clusters[m][n]]] = [r, g, b]    
        self.im = Image.fromarray(self.image)
        self.im.show()
        
        
    def save_file_c_c(self, path):
        """save result for color clustering"""
        self.im.save(path)
    
    
    def save_file_e_(self, path):
        """save result for edge detection"""
        self.ed.save(path)
        

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

