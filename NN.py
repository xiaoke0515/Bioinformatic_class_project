from sparse import NNDataReader
import tensorflow as tf
import numpy as np
import pickle

class NN:
    def __init__(self, classreader, filename, train_proportion=0.7):
        #tf.device('/cpu:0')
        label = np.array(classreader.sample_label)
        self.class_reader = classreader
        self.filename = filename
        self.ChooseData(label, filename, train_proportion)
        #sess = self.BuildGraph()

    def ChooseData (self, label, filename, train_proportion=0.7):
        self.__nn_reader = NNDataReader (filename, batch_size=100, delimiter='\t')
        if not self.__nn_reader.HaveResult():
            self.__nn_reader.Reform(filename, label, train_proportion)
        self.__nn_reader.GetData()
        #print (type(used_data))

    def BuildGraph (self, neuron_number1, neuron_number2, neuron_number3):
        def AddFcLayer (input_neuron, output_neuron, input_variable, layer_num, norm, mean_w=0, mean_b=0, var_w=1, var_b=1):
            w = tf.Variable(tf.truncated_normal(shape=[input_neuron, output_neuron], mean=0.0, stddev=0.1, dtype=tf.float32), trainable=True, name='w' + str(layer_num))
            b = tf.Variable(tf.truncated_normal(shape=[1, output_neuron], mean=0.1, stddev=0.01, dtype=tf.float32), trainable=True, name='b' + str(layer_num))
            out_hat = tf.add(tf.matmul(input_variable, w), b, name='out_hat' + str(layer_num))
            out = tf.nn.relu(out_hat, name='out' + str(layer_num))
            decay = 1e0
            if norm == 1:
                #reg = tf.contrib.layers.l1_regularizer(0.1)(w) + tf.contrib.layers.l1_regularizer(0.1)(b)
                reg =  decay * (tf.reduce_mean(tf.abs(w)) + tf.reduce_mean(tf.abs(b)))
            elif norm == 2:
                #reg = tf.contrib.layers.l2_regularizer(0.1)(w) + tf.contrib.layers.l2_regularizer(0.1)(b)
                reg = decay * (tf.sqrt(tf.reduce_mean(tf.square(w))) + tf.sqrt(tf.reduce_mean(tf.square(b))))
            else:
                reg = 0
            return out, reg
        input_feature_number = self.__nn_reader.gene_number
        output_label_number = self.class_reader.type_number.shape[0]
        neuron_number = [neuron_number1, neuron_number2, neuron_number3 ]
        # place holder
        input_data = tf.placeholder(shape=[None, input_feature_number], dtype=tf.float32, name='input')
        label = tf.placeholder(shape=[None, output_label_number], dtype=tf.float32, name='label')
        reg = 0
        # layer 0
        (out0, reg0) = AddFcLayer(input_feature_number, neuron_number[0], input_data, 0, 2)
        reg += reg0
        # layer 1
        (out1, reg1) = AddFcLayer(neuron_number[0], neuron_number[1], out0, 1, 1)
        reg += reg1
        # layer 2
        (out2, reg2) = AddFcLayer(neuron_number[1], neuron_number[2], out1, 2, 1)
        reg += reg2
        # layer 3
        (out3, reg3) = AddFcLayer(neuron_number[2], output_label_number, out2, 3, 1)
        reg += reg3
        # output
        output = tf.nn.softmax(out3)
        #output = out3
        # loss
        #loss = tf.reduce_mean (tf.square(tf.reduce_sum( label * tf.log(1 + tf.exp(-output)) + (1 - label) * tf.log(1 + tf.exp(output)), reduction_indices=[1])))
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))
        loss = tf.reduce_mean(tf.reduce_sum(- label * tf.log (output), 1))
        #loss = tf.reduce_mean (tf.reduce_sum( - label * tf.log(output) - (1 - label) * tf.log(1 - output), reduction_indices=[1]))
        loss += reg
        # accuracy
        predict_result = tf.equal (tf.argmax(output, axis=1), tf.argmax(label, axis=1))
        accuracy = tf.reduce_mean (tf.cast (predict_result, tf.float32))
        #tf.device('/cpu:0')
        sess = tf.Session()#
        #sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
        #sess.run (tf.global_variables_initializer())
        #print (sess.run (reg))
        #input()
        #print(reg.name)
        self.reg = reg.name
        return sess, accuracy, loss, input_data, label

    def Train(self, sess, accuracy, loss, input_data, label, batch_size = 100, train_time=2000, lr=1e-4):
        #global_step = tf.Variable(0, name='global_step', trainable=False)
        #learning_rate = tf.train.exponential_decay( learning_rate=lr, global_step=global_step, decay_steps=1, decay_rate=10**(-1 / train_time), staircase=False)
        #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses = []
        accuracyes = []
        losses_test = []
        accuracyes_test = []
        graph = tf.get_default_graph()
        reg = graph.get_tensor_by_name(self.reg)
        #w1 = graph.get_tensor_by_name('w0:0')
        #print (sess.run(reg))
        for i in range (train_time):
            (data_, label_) = self.__nn_reader.GetBatch_train()
            #print (sess.run (w1))
            #print (data_)
            #print (np.where (data_ <0.0001))
            #print (label_)
            sess.run(train_step, feed_dict={ input_data:data_, label:label_ })
            #input()
            if i % 10 == 0:
                print (sess.run(reg))
                loss_ = sess.run (loss, feed_dict={ input_data:data_, label:label_ })
                accuracy_ = sess.run (accuracy, feed_dict={ input_data:data_, label:label_ })
                (data_test, label_test) = self.__nn_reader.GetBatch_test()
                loss_test = sess.run (loss, feed_dict={ input_data:data_test, label:label_test })
                accuracy_test = sess.run (accuracy, feed_dict={ input_data:data_test, label:label_test })

                print ('step : ', i, 'loss : ', loss_, ' accuracy : ', accuracy_, ' test: loss : ', loss_test, ' accuracy : ', accuracy_test)
                saver.save(sess, './NN_Checkpoints/Chkp1', global_step=i)
                #print (sess.run(self.reg))
                #print (sess.run (w1))
                losses.append(loss_)
                accuracyes.append (accuracy_)
                losses_test.append(loss_test)
                accuracyes_test.append(accuracy_test)
        picklefile = open('./NN_Checkpoints/losses', 'wb')
        pickle.dump([losses, accuracyes, losses_test, accuracyes_test], picklefile)
        picklefile.close()

    def Test(self, sess, accuracy, loss, input_data, label, batch_size=100):
        saver = tf.train.Saver()
        saver.restore (sess,  tf.train.latest_checkpoint('./NN_Checkpoints/'))
        (data_, label_) = self.__nn_reader.GetBatch_test()
        loss_ = sess.run (loss, feed_dict={ input_data:data_, label:label_ })
        accuracy_ = sess.run (accuracy, feed_dict={ input_data:data_, label:label_ })
        (data_train, label_train) = self.__nn_reader.GetBatch_train()
        loss_train = sess.run (loss, feed_dict={ input_data:data_train, label:label_train })
        accuracy_train = sess.run (accuracy, feed_dict={ input_data:data_train, label:label_train })
        print ('test result: loss : ', loss_, 'accuracy: ', accuracy_, ', train loss :',  loss_train, ' train accuracy: ', accuracy_train)

        print ("_train dataset _____________:")
        for i in range(self.class_reader.type_number.shape[0]):
            samp_data = self.__nn_reader.GetTrain_SameLabel(i)
            labels = np.zeros ([samp_data.shape[0], self.class_reader.type_number.shape[0]])
            labels[:, i] = 1
        
            loss_ = sess.run (loss, feed_dict={ input_data:samp_data, label:labels })
            accuracy_ = sess.run (accuracy, feed_dict={ input_data:samp_data, label:labels })
            print ('type ', i, ' test result: loss : ', loss_, 'accuracy: ', accuracy_)


            
        
        print ("_test dataset _____________:")
        for i in range(self.class_reader.type_number.shape[0]):
            samp_data = self.__nn_reader.GetTest_SameLabel(i)
            labels = np.zeros ([samp_data.shape[0], self.class_reader.type_number.shape[0]])
            labels[:, i] = 1
        
            loss_ = sess.run (loss, feed_dict={ input_data:samp_data, label:labels })
            accuracy_ = sess.run (accuracy, feed_dict={ input_data:samp_data, label:labels })
            print ('type ', i, ' test result: loss : ', loss_, 'accuracy: ', accuracy_)


        
