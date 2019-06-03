#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:54:25 2019

@author: lorenzo
"""

import os
#path = "/Users/lorenzo/Documents/ML/project"
#os.chdir(path)
import random
import sys
sys.path.append('..')
import time
import numpy as np
import pandas as pd

import tensorflow as tf

#import utils

class LSTM:
    
    # In[356]:
    def __init__(self, x1, y1,x2,y2):     
        self.input = x1                
        self.y = y1
        self.input_valid = x2
        self.y_valid = y2
        
        tf.reset_default_graph()
        
        n_steps = 13
        n_inputs = 1
        n_neurons = 150
        n_outputs = 3
        
        #out_weights=tf.Variable(tf.random_normal([n_neurons,n_outputs]))
        #out_bias=tf.Variable(tf.random_normal([n_outputs]))
        
        learning_rate = 0.001
        
        X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        y = tf.placeholder(tf.int32, [None])
        #X=tf.unstack(X ,n_steps,1)
        #X = tf.split(X,n_inputs,1)
        basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
        
        logits = tf.layers.dense(states[-1], n_outputs)
        #logits = tf.matmul(outputs[-1],out_weights)+out_bias
        #logits = tf.nn.xw_plus_b(outputs[-1], out_weights, out_bias)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy)
        loss_summary = tf.summary.scalar('log_loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
        
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        X_train = np.array(self.input)
        X_test = np.array(self.input_valid)
        y_train = np.array(self.y)
        y_test = np.array(self.y_valid)
        
        X_train = X_train.astype(np.float32).reshape(-1, n_steps*n_inputs, 1) 
        
        X_test = X_test.astype(np.float32).reshape(-1, n_steps*n_inputs, 1)
        
        X_valid, X_train = X_train[:5000], X_train[5000:]
        y_valid, y_train = y_train[:5000], y_train[5000:]
        
        X_valid = X_valid.astype(np.float32).reshape(-1, n_steps, n_inputs) 
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
        y_valid = np.array(y_valid).ravel()
        
        def shuffle_batch(X, y, batch_size):
            rnd_idx = np.random.permutation(len(X))
            n_batches = len(X) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                yield X_batch, y_batch
        
        X_test = X_test.reshape((-1, n_steps, n_inputs))
        
        n_epochs = 100
        batch_size = 50
        p = 5
        
        writer = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/test',tf.get_default_graph())
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                    X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
                acc_val = accuracy.eval(feed_dict= {X:X_valid, y:y_valid}) 
                loss_val = loss.eval(feed_dict= {X:X_valid, y:y_valid})
                print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val, 'Loss accuracy:', loss_val)
            m,n=sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_train,y:y_train})
            m1,n1=sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_valid,y:y_valid})
            writer.add_summary(m,p)
            writer.add_summary(n,p)
            test_writer.add_summary(m1,p)
            test_writer.add_summary(n1,p)
            writer.close()
            test_writer.close()
        
                
        with tf.Session() as sess:
            saver.restore(sess, "./my_lstm_model")
            X_new_scaled = X_test
            Z = logits.eval(feed_dict = {X: X_new_scaled})
            y_pred = np.argmax(Z, axis = 1)
        
        print("Predicted classes:", y_pred)
        print("Actual classes:   ", y_test)
