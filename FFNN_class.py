#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

# In[355]:

class FFNN:
    
    # In[356]:
    def __init__(self, x1, y1,x2,y2):     
        self.input = x1                
        self.y = y1
        self.input_valid = x2
        self.y_valid = y2



        tf.reset_default_graph() 
        n_inputs = 13
        n_outputs = 3
        n_hidden1 = 200
        n_hidden2 = 100
        learning_rate = 0.01
        
        
        # In[357]:
        
        
        x = tf.placeholder(shape=(None, n_inputs), dtype=tf.float32, name='x')  
        y = tf.placeholder(shape=(None), dtype=tf.int32, name='y')
        
        
        # In[358]:
        
        
        with tf.name_scope("nn"):
            hidden1 = tf.layers.dense(x, n_hidden1, name="hidden1",
                                      activation=tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                      activation=tf.nn.sigmoid)
            logits = tf.layers.dense(hidden2, n_outputs, name="output1")
        
        
        # In[359]:
        
        
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('log_loss', loss)
        
        
        # In[360]:
        
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)
        
        
        # In[361]:
        
        
        y = tf.cast(y, tf.int32)
        
        
        # In[362]:
           
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
            
        init = tf.global_variables_initializer()
        #saver = tf.train.Saver()
        
        
        # In[364]:
        
        
        n_epochs = 20
        batch_size = 800
        p = 5
        
        
        writer = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/test',tf.get_default_graph())
        
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                a = len(self.input)//batch_size
                for iteration in range(a):
                    idx = batch_size * iteration
                    X_batch = self.input[idx:idx + batch_size].values
                    y_batch_sub = self.y[idx:idx + batch_size].values
                    y_batch = np.asarray([j for i in y_batch_sub for j in i])
                    sess.run(training_op, feed_dict = {x:X_batch, y:y_batch})
                acc_train = accuracy.eval(feed_dict = {x:X_batch, y:y_batch})
                acc_val = accuracy.eval(feed_dict= {x:self.input_valid, y:self.y_valid}) 
                loss_val = loss.eval(feed_dict= {x:self.input_valid, y:self.y_valid})
                print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val, 'Loss accuracy:', loss_val)
            m,n=sess.run([accuracy_summary,loss_summary],feed_dict = {x:self.input,y:self.y.values.ravel()})
            m1,n1=sess.run([accuracy_summary,loss_summary],feed_dict = {x:self.input_valid,y:self.y_valid})
            writer.add_summary(m,p)
            writer.add_summary(n,p)
            test_writer.add_summary(m1,p)
            test_writer.add_summary(n1,p)
            writer.close()
            test_writer.close()


# In[371]:



#with tf.Session() as sess:
#    init.run()
#    for epoch in range(n_epochs):
#        a = len(X_train)//batch_size
#        for iteration in range(a):
#            idx = batch_size * iteration
#            X_batch = X_train[idx:idx + batch_size].to_numpy()
#            y_batch_sub = y_train[idx:idx + batch_size].to_numpy()
#            y_batch = np.asarray([j for i in y_batch_sub for j in i])
#            sess.run( training_op, feed_dict={x:X_batch, y:y_batch} )
#        acc_train = accuracy.eval(feed_dict = {x:X_batch, y:y_batch})
#        acc_val = accuracy.eval(feed_dict= {x:X_valid, y:y_valid})    
#        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
#    save_path = saver.save(sess, "./my_model.ckpt")
#
#
## In[366]:
#
#
#with tf.Session() as sess:
#    saver.restore(sess, "./my_model.ckpt")
#    Z = logits.eval(feed_dict = {x: X_valid})
#    y_pred = np.argmax(Z, axis = 1)
#
#print("Predicted classes:", y_pred)
#print("Actual classes:   ", y_valid)


# In[367]:


#set(y_pred)
#
#
## In[368]:
#
#
#y_pred = pd.DataFrame(y_pred)
#
#
## In[369]:
#
#
#y_pred.columns = ['label']
#
#
## In[372]:
#
#
#sum(y_pred.label)


# In[ ]:





# In[ ]:

    
