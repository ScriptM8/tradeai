import random
import time
import datetime
import BinanceAPI
from livelossplot import PlotLossesKerasTF
class Helper():
    API_KEY = "kQBtmUMTAKJdJMocaK8zIoY4dCiKvmySGLwhKhqHekqLhEzfEwtC6uEouzmnDTyo"
    API_SECRET = "DR97PHTJ8sEMmRJHo94muqWG0TwwRSg7wNwejvbdI5ljxO43SK2g2rQN8ualXquc"
    instance = ""
    result = []
    timeframes = {
        "local":50,
        "middle":170,
        "global":500
    }
    infoArr = []
    startFrom = timeframes["global"]

    def __init__(self):
        self.connect(self.API_KEY, self.API_SECRET)

    def connect(self, API_KEY, API_SECRET):
        self.instance = BinanceAPI.Binance(API_KEY, API_SECRET)
        return self.instance

    def convertToAIFormat(self, data):
        list = []
        arr = []
        for item in data:
            list.append({
                "START_TIME": item[0],
                "OPEN": float(item[1]),
                "HIGH": float(item[2]),
                "LOW": float(item[3]),
                "CLOSE": float(item[4]),
                "VOLUME": float(item[5]),
                "TAKER_BASSET_VOLUME": float(item[9]),
                "TAKER_QASSET_VOLUME": float(item[10]),
                "local": {
                    "RES": 0,
                    "SUP": 0
                },
                "middle": {
                    "RES": 0,
                    "SUP": 0
                },
                "global": {
                    "RES": 0,
                    "SUP": 0
                },
            })
        
        lines = []
        for index, timeframeName in enumerate(self.timeframes):
          print(self.timeframes[timeframeName])
          for i in range(self.timeframes[timeframeName], len(list)):
            a = 0
            b = 30000
            for j in range(i - self.timeframes[timeframeName], i+1):
              a = max(a, list[j]["CLOSE"])
              b = min(b, list[j]["CLOSE"])
            list[i][timeframeName]["RES"] = a
            list[i][timeframeName]["SUP"] = b
        
        #for index, item in enumerate(list):
          #print(item["RES"], item["SUP"])
          
        #self.result = []
        for index, item in enumerate(list):
          #bar = round((float(item["CLOSE"]) / float(item["OPEN"]) - 1) * 100, 2)
          bar = item["CLOSE"] - item["OPEN"]
          if(bar > 0):
            beginGreenShadow = item["CLOSE"]
            beginRedShadow = item["OPEN"]
          else:
            beginGreenShadow = item["OPEN"]
            beginRedShadow = item["CLOSE"]
          #greenShadow = round((float(item["HIGH"]) / float(item["OPEN"]) - 1) * 100, 2)
          #redShadow = round((float(item["LOW"]) / float(item["OPEN"]) - 1) * 100, 2)

          greenShadow =  item["HIGH"] - beginGreenShadow
          redShadow = -(beginRedShadow - item["LOW"])

          linesIndicators = []
          for index, timeframeName in enumerate(self.timeframes):
            localSupDelta = item["CLOSE"] - item[timeframeName]["SUP"]
            localResDelta = item[timeframeName]["RES"] - item["CLOSE"]

            linesIndicator = 0
            if((localSupDelta + localResDelta != 0) and (localSupDelta + localResDelta != 0)):
              if(localResDelta >= localSupDelta):
                linesIndicator = (localResDelta / (localSupDelta + localResDelta))
              else:
                linesIndicator = 1 - (localSupDelta / (localSupDelta + localResDelta))
          
            #if localResDelta >= localSupDelta:
            #  linesIndicator = (localResDelta / (localSupDelta + localResDelta))
            #else:
            #  linesIndicator = 1 - (localSupDelta / (localSupDelta + localResDelta))

            linesIndicators.append(round(linesIndicator, 2))
          #print(linesIndicators)
          #print(linesIndicator)

          #if(greenShadow != 0):
              #greenShadow = round(bar / greenShadow, 2)
          #else:
              #greenShadow = 0

          #if(redShadow != 0):
              #redShadow = round(bar / redShadow,2)
          #else:
              #redShadow = 0

          #greenShadow =  round(float(item["HIGH"]) - float(item["OPEN"]),2)
          #redShadow = (round(float(item["LOW"]) - float(item["OPEN"]),2))
          
          arr.append([ 
                        round(bar,7),
                        round(greenShadow,7),
                        round(redShadow,7),
                        linesIndicators[0],
                        linesIndicators[1],
                        linesIndicators[2]
                        #round(localResDelta,7),
                        #round(localSupDelta,7),
                        #round(item["VOLUME"],7),
                        #datetime.datetime.fromtimestamp(int(str(item["START_TIME"])[0:-3])).isoformat(),

          ])
          self.infoArr.append([
              round(bar,7),        
              (float(item["CLOSE"]) / float(item["OPEN"]) - 1),
              float(item["OPEN"]),
              float(item["CLOSE"]),
              datetime.datetime.fromtimestamp(int(str(item['START_TIME'])[0:-3])).strftime('%Y-%m-%d-%H')
          ])

        arr = arr[self.startFrom:len(arr)]
        self.infoArr = self.infoArr[self.startFrom:len(self.infoArr)]
        return arr

    #def getSupport(self, bar):
        #localData = []
        #lines = self.supportLines
        #for index, item in enumerate(data):
          #if(localData["local"] < list.):


          
        #self.result
        #self.result = self.convertToAIFormat(data)
    #def getReistance(self, data):
        #self.result = self.convertToAIFormat(data)