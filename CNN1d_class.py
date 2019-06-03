
import pandas as pd
import numpy as np
import os
import tensorflow as tf

#path = "/Users/lorenzo/Documents/ML/project"
#os.chdir(path)

class CNN_1d:
    
    # In[356]:
    def __init__(self, x1, y1,x2,y2):     
        self.input = x1                
        self.y = y1
        self.input_valid = x2
        self.y_valid = y2

        n_inputs = 13
        channels = 15000
                
        conv1_fmaps = 32
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"
        
        conv2_fmaps = 64
        conv2_ksize = 3
        conv2_stride = 1
        conv2_pad = "SAME"
        
        #conv3_fmaps = 128
        #conv3_ksize = 10
        #conv3_stride = 1
        #conv3_pad = "SAME"
        
        #pool3_dropout_rate = 0.25
        pool3_fmaps = conv2_fmaps
        
        # Define a fully connected layer 
        #n_fc1 = 3
        #fc1_dropout_rate = 0.5
        
        # Output
        n_outputs = 3
        
        nrows, ncols = self.input.shape
        #X_train = np.array(X_train)
        #A = X_train.reshape(X_train.shape + (3,))
        
        X_train_array = (np.array(self.input)).reshape(nrows,ncols,-1)
        #y_train_array = np.array(y_train)
        #y_test_array = np.array(y_test)
        n2rows, n2cols = self.input_valid.shape
        X_test_array = (np.array(self.input_valid)).reshape(n2rows,n2cols,-1)
        
        y_train_array = np.array(self.y).ravel()
        y_test_array = np.array(self.y_valid).ravel()
        
        X_valid_array, X_train_array = X_train_array[:5000], X_train_array[5000:]
        y_valid_array, y_train_array = y_train_array[:5000], y_train_array[5000:]
        
        X_train_array = X_train_array.astype(np.float32)
        X_valid_array = X_valid_array.astype(np.float32)
        X_test_array = X_test_array.astype(np.float32)
        y_train_array = y_train_array.astype(np.int32)
        y_valid_array = y_valid_array.astype(np.int32)
        y_test_array = y_test_array.astype(np.int32)
        
        tf.reset_default_graph()
        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, shape=(None,n_inputs,1), name="X")
            
            y = tf.placeholder(tf.int32, shape=[None], name="y")
            training = tf.placeholder_with_default(False, shape=[], name='training')
            
        conv1 = tf.layers.conv1d(X, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                 strides=conv1_stride, padding=conv1_pad,
                                 activation=tf.nn.sigmoid, name="conv1")
        
        conv2 = tf.layers.conv1d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                                 strides=conv2_stride, padding=conv2_pad,
                                 activation=tf.nn.sigmoid, name="conv2")
        
        # conv3 = tf.layers.conv1d(conv2, filters=conv3_fmaps, kernel_size=conv3_ksize,
        #                          strides=conv3_stride, padding=conv3_pad,
        #                          activation=tf.nn.sigmoid, name="conv3")
        # 
        # =============================================================================
        # Step 4: Set up the pooling layer with dropout using tf.nn.max_pool 
        with tf.name_scope("pool3"):
            
            pool3 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding="VALID")
           
        # Step 5: Set up the fully connected layer using tf.layers.dense
        with tf.name_scope("fc1"):
        
            fc1 = tf.reshape(pool3, (-1, pool3_fmaps*6))
            
            #fc1_drop = tf.nn.dropout(fc1, keep_prob=tf.placeholder(tf.float32))
        
        # Step 6: Calculate final output from the output of the fully connected layer
        with tf.name_scope("output"):
            logits = tf.layers.dense(fc1, 3,name="output1")
           
            #Y_proba = tf.nn.softmax(logits, name="Y_proba1")
        
        # Step 5: Define the optimizer; taking as input (learning_rate) and (loss)
        with tf.name_scope("train"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
            
            
        # Step 6: Define the evaluation metric
        with tf.name_scope("eval"):
            
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        # Step 7: Initiate    
        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
        
        
        # Step 9: Define some necessary functions
        def get_model_params():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
        
        def restore_model_params(model_params):
            gvar_names = list(model_params.keys())
            assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                          for gvar_name in gvar_names}
            init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
            feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
            tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
        
        def shuffle_batch(X, y, batch_size):
            rnd_idx = np.random.permutation(len(X))
            n_batches = len(X) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = X[batch_idx], y[batch_idx]
                yield X_batch, y_batch
                
        n_epochs = 50
        iteration = 0
        batch_size = 100
        best_loss_val = np.infty
        check_interval = 500
        checks_since_last_progress = 0
        max_checks_without_progress = 10
        best_model_params = None 
        p=5
        
        writer = tf.summary.FileWriter('./graphs/train_sig_sig', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/valid_sig_sig',tf.get_default_graph())
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
            
                for X_batch, y_batch in shuffle_batch(X_train_array, y_train_array, batch_size):
                    iteration += 1
                    sess.run(training_op, feed_dict={X: X_train_array, y: y_train_array, training: True})
                    if iteration % check_interval == 0:
                        loss_val = loss.eval(feed_dict={X: X_batch, y: y_batch})
                        if loss_val < best_loss_val:
                            best_loss_val = loss_val
                            checks_since_last_progress = 0
                            best_model_params = get_model_params()
                        else:
                            checks_since_last_progress += 1
                acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_val = accuracy.eval(feed_dict={X: X_valid_array, y: y_valid_array})
                print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                          epoch, acc_batch * 100, acc_val * 100, best_loss_val))
                if checks_since_last_progress > max_checks_without_progress:
                    print("Early stopping!")
                    break
        
            if best_model_params:
                restore_model_params(best_model_params)
            acc_test = accuracy.eval(feed_dict={X: X_test_array, y: y_test_array})
            print("Final accuracy on test set:", acc_test)
            save_path = saver.save(sess, "./my_model")
            m,n=sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_train_array,y:y_train_array})
            m1,n1=sess.run([accuracy_summary,loss_summary],feed_dict = {X:X_valid_array,y:y_valid_array})
            writer.add_summary(m,p)
            writer.add_summary(n,p)
            test_writer.add_summary(m1,p)
            test_writer.add_summary(n1,p)
            
        writer.close()
        test_writer.close()
        
        
        with tf.Session() as sess:
            saver.restore(sess, "./my_model")
            X_new_scaled = X_test_array
            Z = logits.eval(feed_dict = {X: X_new_scaled})
            y_pred = np.argmax(Z, axis = 1)
        
        print("Predicted classes:", y_pred)
        print("Actual classes:   ", y_test_array)


