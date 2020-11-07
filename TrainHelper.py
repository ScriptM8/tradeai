import random
import DataHelper
class TrainHelper:
    limit = 10
    count = 500
    mode = "smaller_to_smaller"
    useCustomTests = 0.3
    customTests = []
    customResults = []
    #mode = "smaller_to_larger"
 
    def convertToTrainData(self, data):
        helper = DataHelper.Helper()
        data = helper.convertToAIFormat(data)
        unit_to_multiplier = {
          "smaller_to_smaller": self.smallerToSmaller(data),
          "smaller_to_larger": self.smallerToLarger(data),
        }
        return unit_to_multiplier[self.mode]

    def smallerToSmaller(self, data):
      localArray = []
      tests = []
      results = []
      limit = self.limit
      for index in range(len(data)):
        if(self.limit+index < len(data)):
            localArray = []
            locLimit = 0
            while locLimit < self.limit:
                localArray.append(data[locLimit+index])
                locLimit+=1
            tests.append(localArray)
            r = [int(data[index+(self.limit)][0] < 0), int(data[index+(self.limit)][0] >= 0)] 
            #TODO Refactor result. ->> add for cycle to calculate 
            '''
            Pseudo code
            rDelta = 0
            for i in range(start, start + n):
              rDelta += data[i]
            if( rDelta > 0) --> upTrend
            else --> downTrend
            '''
            #r = data[self.limit+index][0]
            #r = int(data[index+(self.limit)][0] > 0)
            results.append(r)
            limit = self.limit
      begin = int(len(tests) * 0.3) # tut on beret range (0.7:0.8) iz vseh dannih -> budushij test
      end = int(len(tests) * 0.5)
      cnt = end - begin
      #for index,item in enumerate(tests)
      for i in range(cnt):
        self.customTests.append(tests[begin])
        del tests[begin]
        self.customResults.append(DataHelper.Helper.infoArr[begin+TrainHelper.limit])
        del results[begin]
        del DataHelper.Helper.infoArr[begin]
      print(self.customTests)
      print(self.customResults)
      print("yee boii")

      print(data[(-10 - self.limit):-1])
      print(tests[-10:-1])
      print(results[-10:-1])
      #'''
      return {
        "tests": tests,
        "results": results,
    }

    def smallerToLarger(self, data):
      return
      localArray = []
      tests = []
      results = []
      limit = self.limit
      #print(data[:30])
      for index, item in enumerate(data):
            if((self.limit+2)+index < len(data)):
                if(self.limit != 1): 
                    localArray = []
                    locLimit = 0
                    while limit > 0:
                        limit = limit-1
                        localArray.append(data[limit+index])
                    tests.append(localArray)
                else:
                    tests.append(data[index])
                if(index < 10):
                    print(data[index+(self.limit)-1][0],data[index+(self.limit+2)-1][0],data[index+(self.limit+1)-1][0])
                #r = int((data[index+(self.limit+4)-1][0] + data[index+(self.limit+1)-1][0]) < 0) # with the raw deltas
                #r = [int(data[index+(self.limit)][0] <= 0), int(data[index+(self.limit)][0] > 0)]
                r = int(data[index+(self.limit)][0] > 0)
                results.append(r)
                limit = self.limit
      #print(tests[:20])
      #print(results[:10])
      return {
        "tests": tests,
        "results": results,
    }