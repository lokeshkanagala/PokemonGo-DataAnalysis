import random
from datetime import  datetime
from pyspark import SparkContext
from pyspark.sql import Row, SQLContext
import  os,math,re,array, sys
from pyspark.sql.functions import monotonically_increasing_id,udf,col,split

def compareZones(line):
    zone = line[0]
    PredZone = line[1]
    for i in PredZone:
        if zone == i:
            return 1
        else:
            return 0

''' !!!! This function assigns each latitude, longitude pair to its nearest Zone !!!! '''
def DoKmeans(x):
    
    longitude=x[0]
    latitude=x[1]
    k=1
    #Starting the process by taking infinity as the nearest Distance
    nearDist = float("inf")
	
    for cen in ListOfCentroids:
        distance=math.sqrt(math.pow((latitude-cen[0]),2)+math.pow((longitude-cen[1]),2))
        if distance<nearDist:
            nearDist=distance
            Zonec=k
        k+=1
    return [Zonec, [latitude,longitude,x[4]]]


''' !!!! Function to get the time in temrs of hours !!!! '''
def returnHours(Tm):
    TmArray = re.split(":", Tm)
    mins = float(TmArray[1])
    hr = float(TmArray[0])
	#This function converts hour 3:45 to 3.75
    if mins != 0.0:
        timeInHours = mins / 60.0 + hr
    else:
        timeInHours = hr
    return timeInHours
	
'''  !!!! Function to create 24 time periods !!!! '''
def returnPeriod(p):
    t24 = datetime.strptime(p,'%H:%M:%S').strftime('%X')
    temp=int(t24[0:2])
    period=temp+1
    return period
	
''' !!!! Function to find the centroid of each zone !!!! '''
def FindCentroid(z):

    sumOfLat=0.0
    sumOfLong=0.0
    ZoneNum=z[0]
    LatLongPairs=list(z[1])
    count=0.0
    for i in LatLongPairs:
        count+=1
        sumOfLat=sumOfLat+i[0]
        sumOfLong=sumOfLong+i[1]
    newLat=sumOfLat/count
    newLong=sumOfLong/count
    return [ZoneNum,[newLat,newLong]]
	

if __name__ == "__main__":

    sc=SparkContext()
    sqlContext=SQLContext(sc)
    inputData = sys.argv[1]
    df = sqlContext.read.load(inputData, format='com.databricks.spark.csv', header='true', inferSchema='true')
    df = df.filter(df.appearedLocalTime.isNotNull())
    #Filtering the csv file to discard any bad record
    split_col = split(df['appearedLocalTime'], 'T')
    df = df.withColumn('AppearedTime', split_col.getItem(1)).withColumn('AppearedDay', split_col.getItem(0))
	
    sp=udf(lambda x:datetime.strptime(x,'%Y-%M-%d').strftime('%A'))
    hoursUDF = udf(returnHours)
    periodUDF = udf(returnPeriod)
    totaldata = df.withColumn('Weekday', sp(col('AppearedDay'))).withColumn("hour", hoursUDF(col('AppearedTime'))).withColumn("id",monotonically_increasing_id()).withColumn("Period", periodUDF(col("AppearedTime")))
    KMDF=totaldata.select('longitude','latitude','hour','Weekday','id', 'Period', 'weather')
    KMDF.registerTempTable('KMTable')
    
    '''!!!!!!!!!! This block of code implements Kmeans !!!!!!!!!!!!'''
    
    LatCentroid=sqlContext.sql("SELECT latitude from KMTable").collect()
    LongCentroid=sqlContext.sql("SELECT longitude from KMTable").collect()
    LatC=LatCentroid[:100]
    LongC=LongCentroid[:100]
	
    #Dividing the total area into 100 Zones and finding the Zone in which each reocrd falls in 
	
    centroids=zip(LatC,LongC)
    ListOfCentroids=[]
    
    #Taking the centroid list into a list, for the ease of iteration
    for cen in centroids:
        rec=[]
        xx=cen[0].latitude
        yy=cen[1].longitude
        rec=[xx,yy]
        ListOfCentroids.append(rec)
     
    for itr in range(15):
        TempRdd=KMDF.rdd.map(lambda x:DoKmeans(x))
        newRdd=TempRdd.groupByKey().map(FindCentroid)
        cRdd=newRdd.collect()
        for eachh in cRdd:
            zonenum=eachh[0]
            newcen=eachh[1]
            ListOfCentroids[zonenum-1][0]=newcen[0]
            ListOfCentroids[zonenum-1][1]=newcen[1]

    classrow=TempRdd.map(lambda x: (Row(Zone=x[0],idxx=x[1][2])))
    finaldf=sqlContext.createDataFrame(classrow)
	
    #Adding a column called "Zone" to each record/row in the RDD
    NBDF=KMDF.join(finaldf,(KMDF.id==finaldf.idxx))
    NBDF.registerTempTable('NBtable')
    sqlContext.cacheTable("NBtable")
    
    ''' This block of code implements the Naive-Bayes '''

    #To get the total vocabulary of Periods in the file
    PeriodVocabulary=sqlContext.sql("SELECT count(distinct(Period)) from NBtable").collect()[0][0]
    #To get the total vocabulary of days in the file
    DayVocabulary=sqlContext.sql("SELECT count(distinct(Weekday)) from NBtable").collect()[0][0]
    #To get the total vocabulary of weather in the file
    weatherVocabulary=sqlContext.sql("SELECT count(distinct(weather)) from NBtable").collect()[0][0]
    totaltimes=sqlContext.sql("SELECT Period FROM NBtable").collect()
    weeklist=sqlContext.sql("SELECT Weekday FROM NBtable").collect()
    
    alp=[0.001,0.01,2.0,3.5,5.0,10.0,20.0]
    accuracies=[]
    arrayy = []
    Result = {}
    listings=[]
    
    for alpha in alp:
    	for i in range(len(totaltimes)):
        	for zone in range(1,101):
        		zne = sqlContext.sql("SELECT Zone FROM NBtable WHERE Zone='" + str(zone) + "'").collect()
        		priorprobZone = float(len(zne)) / len(totaltimes)
        		occP = sqlContext.sql(
            		"SELECT Period FROM NBtable WHERE Zone='" + str(zone) + "' AND Period='" + totaltimes[i].Period + "'").collect()
            
			''' Finding number of occurences of given period in each zone'''
        		probPeriod = (float(len(occP)) + alpha) / float(len(zne) + alpha * PeriodVocabulary)
        	
        	    occWe = sqlContext.sql(
            		"SELECT Period FROM NBtable WHERE Zone='" + str(zone) + "' AND Weather='" + totaltimes[i].weather + "'").collect()
                probWeather = (float(len(occWe)) + alpha) / float(len(zne) + alpha * WeatherVocabulary)
        	
        		occW = sqlContext.sql(
            		"SELECT Weekday FROM NBtable WHERE Zone='" + str(zone) + "' AND Weekday='" + weeklist[i].Weekday+ "'").collect()
			
			''' Finding number of occurences of given weekday in each zone'''
        		probWeekday = (float(len(occW)) + alpha) / float(len(zne) + alpha * DayVocabulary)
        		Probability = priorprobZone * probPeriod * probWeekday
        		Result[zone] = Probability
        	
       		maxprob = max(Result.values())
    		maxzone = [x for x, y in Result.items() if y == maxprob]
    
    	  	arrayy.append(maxzone)
                tempp=i
        	listings.append(tempp)
        	zoneid=zip(listings,arrayy)
        
        predictedZones=sc.parallelize(zoneid).map(lambda x: (Row(idrow=x[0],zonePredicted=x[1])))
        DFF = sqlContext.createDataFrame(predictedZones)
        NBDFFinal=NBDF.join(DFF, (NBDF.id == DFF.idrow))

        similarity = NBDFFinal.select('Zone', 'zonePredicted').rdd.map(compareZones).filter(lambda x: x ==1).count()
    
        '''Calculating the accuracies of NaivebAyes implementation'''
    
        acc = float(similarity)/float(NBDFFinal.rdd.count()) * 100
        ResultAccuracies.append(acc)
    print ResultAccuracies
