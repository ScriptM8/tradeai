from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys, os
import math
import DataHelper
import TrainHelper
import BinanceAPI
import time
import pylab as pl
from IPython import display
from livelossplot import PlotLossesKerasTF
#tf.logging.set_verbosity(tf.logging.ERROR)

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr
    new_lr = old_lr * 0.999
    print("\nEpoch: {}. changing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr = new_lr

class TensorFlow():
    data = ""
    class_names = [
        1,  # GREEN
        0,  # RED
    ]
    depth = 6
    epochs = 150
    batch_size = 150
    steps_per_epoch = 50
    def __init__(self, data):
        self.data = data

    def createModel(self):
        layersArray = []
        inputArrays = []
        Nodes = 128
        countBarLayers = 3 #hp.Int('count_bar_layers', min_value=1, max_value=70, step=10)
        unitsCountBars = 3 #hp.Int('units_input_bars', min_value=1, max_value=128, step=8)
        
        activationMain = 'tanh'#hp.Choice('activation_main', values=['tanh', 'relu'])

        _regularizers = 0 #hp.Float('regularizers', min_value=0.00001, max_value=0.001, step=0.00001)
        
        _lr = 0.01
  
        _dropout = 0 #hp.Float('dropouts', min_value=0, max_value=0.5, step=0.05)
        training = True #hp.Boolean('training')

        optim = 2 #hp.Int('optimizer', min_value=1, max_value=3)
        # 1 - Adam
        # 2 - SGD
        # 3 - RMSprop

        if(optim == 1):
          optim = tf.keras.optimizers.Adam(learning_rate=_lr) 
        else:
          if(optim == 2):
            optim = tf.keras.optimizers.SGD(learning_rate=_lr)
          else: 
            optim = tf.keras.optimizers.RMSprop(learning_rate=_lr)

        for layerNumber in range(TrainHelper.TrainHelper.limit):
          x = tf.keras.layers.Input(shape=(1,self.depth), dtype=tf.float32)
          inputArrays.append(x)
          for i in range(countBarLayers):
            x = tf.keras.layers.Dense(Nodes, activation=activationMain, kernel_regularizer=tf.keras.regularizers.l2(_regularizers))(x)
            x = tf.keras.layers.Dropout(_dropout)(x, training=True)
          layersArray.append(x)
        merged = tf.keras.layers.Concatenate(axis=1)(layersArray)  

        dense = tf.keras.layers.Dense(Nodes, activation=activationMain)(merged)
        dense = tf.keras.layers.Dropout(_dropout)(dense, training=True)
        for i in range(unitsCountBars):
          dense = tf.keras.layers.Dense(Nodes, activation=activationMain, kernel_regularizer=tf.keras.regularizers.l2(_lr))(dense)
          dense = tf.keras.layers.Dropout(_dropout)(dense, training=True)
        pooling = tf.keras.layers.GlobalAveragePooling1D()(dense)
        output = tf.keras.layers.Dense(2, activation='softmax')(pooling)
        print(inputArrays)
        print(output)
        self.model = tf.keras.models.Model(inputArrays, output)
        self.model.compile(
                      loss="categorical_crossentropy",
                      optimizer=optim,
                      #optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=['accuracy']
                      )
        
        self.model.summary()
        return self.model

    def afterTune(self):
        tmpModel = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        tmpModel.summary()
        tmpModel.trainable = True
        x = tf.keras.layers.Conv1D(TrainHelper.TrainHelper.limit, self.depth-1, activation='tanh')(tmpModel.layers[-1].output, training=True)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(TrainHelper.TrainHelper.limit, self.depth-1, activation='tanh')(x, training=True)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(2,activation='softmax')(x)
        self.model = tf.keras.Model(inputs = tmpModel.input, outputs = [x])
        self.model.summary()
        self.model.compile(
                      loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.SGD(1e-2),
                      metrics=['accuracy']
                      )
        self.trainModel()
        for layer in self.model.layers[:int(len(self.model.layers)*2/3)]:
          layer.trainable = False
        self.model.compile(
                      loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.SGD(1e-4),
                      metrics=['accuracy']
                      )
        self.model.summary()
        self.trainModel()
        
        

    def trainModel(self):
        print(self.normalizeTests(self.data["tests"]))
        print(len(self.data["results"]))
        self.batch_size = int((len(self.data["results"]) / self.epochs))
        print(f'batch_size: {self.batch_size}')
        callbacks = [
          LearningRateReducerCb(),
          tf.keras.callbacks.ModelCheckpoint('C:/Users/MAXIME/OneDrive/Documents/DebugVS', monitor='accuracy', mode='auto',save_best_only=True, verbose=0),
          PlotLossesKerasTF(),
        ]
        self.history = self.model.fit(
            #self.dataset(self.data["tests"], self.data["results"]),
            self.normalizeTests(self.data["tests"]),
            self.normalizeResults(self.data["results"]),
            #tf.convert_to_tensor(self.data["tests"], dtype=tf.float32),
            #tf.convert_to_tensor(self.data["results"], dtype=tf.float32),
            validation_split=0.13, #0.13 works good
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            verbose=0,
            shuffle=True,
            callbacks=callbacks,
        )

    def normalizeTests(self, data):
        #print(data[:TrainHelper.limit])
        tests = []
        result = []
        localTest = dict()
        for i in range(TrainHelper.TrainHelper.limit):
          localTest[i] = []
        for indexTest, test in enumerate(data):
          for indexArrayItem, arrayItem in enumerate(test):
            localTest[indexArrayItem].append([arrayItem])

        #for indexTest, test in enumerate(localTest):
        #  result.append(test)
        #for i in range(TrainHelper.limit):
          #print(localTest[i])
        #print('hehe')
        ansArr = []
        for i in range(TrainHelper.TrainHelper.limit):
          ansArr.append(tf.convert_to_tensor(localTest[i], dtype=tf.float32))
        #print(ansArr)
        return ansArr
    def normalizeResults(self, data):
        print(tf.convert_to_tensor(data, dtype=tf.float32))
        return tf.convert_to_tensor(data, dtype=tf.float32)
    def makePrediction(self, tryMe):
        a = self.model.predict(tryMe)
        print('using last epoch Model: ')
        if float(a[0][0]) >= float(a[0][1]):
          print('Short positions ' + str(a[0][0]) + '%')
        else:
          print('Long positions ' + str(a[0][1]) + '%')
        
    def predictFromSave(self, tryMe):
        print('')
        print('using best accurate Model')
        bestModel = tf.keras.models.load_model('/tmp')
        b = bestModel.predict(tryMe)
        if float(b[0][0]) >= float(b[0][1]):
          print('Short positions ' + str(b[0][0]) + '%')
        else:
          print('Long positions ' + str(b[0][1]) + '%')
        
    def printModel(self):
        history = self.history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.title('Training and Validation Accuracy')

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.show()
    
    def emulateTrading(self):
        class Pos:
          def __init__(self, entry, volume):
            openPosesDeltaStart = 0.005
            openPosesDelta = 0.0003
            self.entry = entry
            self.volume = volume
            if(len(openPos)):
              self.closeDelta = openPosesDeltaStart + (openPosesDelta * len(openPos))
            else:
              self.closeDelta = openPosesDeltaStart
        openPos = []
        class SPos:
          def __init__(self, entry, volume):
            openPosesDeltaStart = 0.005
            openPosesDelta = 0.0003
            self.entry = entry
            self.volume = volume
            if(len(openSPos)):
              self.closeDelta = openPosesDeltaStart + (openPosesDelta * len(openSPos))
            else:
              self.closeDelta = openPosesDeltaStart
        openSPos = []
        self.balance = 1000
        self.percentOnBet = 0.03
        self.percentOnBetDelta = 0.03
        self.fee = 0.05/100 #futures fee
        self.delta = 0
        self.profit = 0
        predictionPercent = 0.75
        difference = 0.05
        predHigh = 1
        stats = {
            "lose": 0,
            "win": 0
        }
        totalWin = 0
        totalFee = 0
        totalLose = 0
        
        #newBalance = balance-(balance*percentOnBet) + (delta*(balance*percentOnBet))
        bestModel = tf.keras.models.load_model('/tmp')
        accuracy = 100
        
        keys_ar = [0]
        balance_ar = [self.balance]
        coin_cost_ar = [TrainHelper.TrainHelper.customResults[0][3]]
        #coin_keys_ar = [TrainHelper.customResults[0][4]]

        def calcDelta(pos, cur):
            return ( cur / pos - 1)
        def curVolume(dir):
            if (dir):
              return self.balance * (self.percentOnBet + (self.percentOnBetDelta * (len(openPos) + 1)))
            else:
              return self.balance * (self.percentOnBet + (self.percentOnBetDelta * (len(openSPos) + 1)))
        def calcFee(vol):
            return self.fee * (vol)
        def makeTrade(dir):
            if (dir):
              tmp = (curVolume(True))
            else:
              tmp = (curVolume(False))
            self.balance -= ( tmp + calcFee(tmp)  ) 
            return tmp
        def takeProfit(pos, vol, cur, tip):
            self.balance += vol + calcDelta(pos, cur) * vol * tip - calcFee(vol)
        #self.updateChart(keys_ar, balance_ar, win_ar, lose_ar, fee_ar, stats, len(openPos))


        #BackTesting
        for index, item in enumerate(TrainHelper.TrainHelper.customTests):
          isOpenPos = False
          currentPrice = TrainHelper.TrainHelper.customResults[index][3]
          '''
          0 - delta
          1 - delta%
          2 - open
          3 - close
          4 - time open
          '''
          #self.delta = TrainHelper.customResults[index][1]
          pred = bestModel.predict(self.normalizeTests(tf.convert_to_tensor([TrainHelper.TrainHelper.customTests[index]],dtype=tf.float32)))
          #pred[0][0] => short pred[0][1] => long %

          #********************************* Iterate section
          #1 Check long positions // mozno kone4no sdelatj 1 klass i booleans type = short ? long (nado podumatj kak ly4we)
          for i in openPos:
            posPrice = i.entry
            posVolume = i.volume
            closeDelta = i.closeDelta
            #TODO implement stop-loss ( if needed )
            #Profit ->
            #1.1 variant manually take'
            print(calcDelta(posPrice, currentPrice))
            print(closeDelta)
            '''
            if (calcDelta(posPrice, currentPrice) >= closeDelta):
                takeProfit(posPrice, posVolume, currentPrice, 1)
                openPos.remove(i)
            '''
            #openPosesTotatlVolume = 

            #1.2 variant, take profit, when NN signals about the change of trend ( and we are in winnig position).
            
            if (pred[0][0] >= predictionPercent and calcDelta(posPrice, currentPrice) >= closeDelta):
                takeProfit(posPrice, posVolume, currentPrice, 1)
                openPos.remove(i)
            
          #2 Iterate over short positions
          for i in openSPos:
            posPrice = i.entry
            posVolume = i.volume
            closeDelta = i.closeDelta
            #2.1 take constant profit
            '''
            if (calcDelta(posPrice, currentPrice)*-1 >= closeDelta):
                takeProfit(posPrice, posVolume, currentPrice, -1)
                openSPos.remove(i)
            ''' 
            #2.2. take profit when not sure to continue 
            
            if (pred[0][1] >= predictionPercent and calcDelta(posPrice, currentPrice)*-1 >= closeDelta):
                takeProfit(posPrice, posVolume, currentPrice, -1)
                openSPos.remove(i)

          
          
          
          #******************************************* Order section
          #3 buy ( long only )
          
          if ((pred[0][1] >= predictionPercent) and ((len(openPos) == 0) or (calcDelta(openPos[-1].entry, currentPrice) < (difference * -1)))):
            openPos.append(Pos(TrainHelper.TrainHelper.customResults[index][3], makeTrade(True)))
          
          
          #4 buy Short
          if(pred[0][0] >= predictionPercent and ((len(openSPos) == 0) or (calcDelta(openSPos[-1].entry, currentPrice) >= (difference)))):
            openSPos.append(SPos(TrainHelper.TrainHelper.customResults[index][3], makeTrade(False)))
          
          ''' TODO how many to open???
          elif (pred[0][1] >= 0.8)
            for i in range(0, 2):
              openPos.append(Pos(TrainHelper.customResults[index][3], volume()))
          '''



          #******************************************** Graph section
          keys_ar.append(index+1)
          balance_ar.append(self.balance)
          
          #coin_keys_ar.append(TrainHelper.customResults[index][4])
          coin_cost_ar.append(TrainHelper.TrainHelper.customResults[index][3])
          
          #if(isOpenPos):
          self.updateChart(keys_ar, balance_ar, stats, len(openPos), coin_cost_ar, len(openSPos))

    def updateChart(self, keys, total_data, stats, cnt, coin_cost, cnt2):
        plt.ion()

        display.clear_output(wait=True)

        fg = plt.figure(figsize=(5, 3), constrained_layout=True)
        gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fg)
        fig_ax_1 = fg.add_subplot(gs[0, 0])
        plt.plot(keys, total_data, label='Total Balance')
        fig_ax_2 = fg.add_subplot(gs[0, 1])
        plt.plot(keys, coin_cost, label='Coin cost')

        print(f'Stats w/l: {stats["win"]}/{stats["lose"]}')
        if(stats["win"] > 0):
          print(f'Winrate: {round(stats["win"]/(stats["win"]+stats["lose"]) * 100,2)}%')
        print(total_data[-1])
        print(f'count of open pos: {cnt}')
        print(f'count of open Spos: {cnt2}')
        
        #plt.plot(keys, win_data, label='Win Data')
        #plt.plot(keys, lose_data, label='Lose Data')
        #plt.plot(keys, fee_data, label='Fee Data')
        plt.draw()

        plt.ioff()
        plt.show()
        print(self.profit)

