// This is the Sentinel-2 collection (all the possible available Sentinel 2 imagery)
var S2_collection = ee.ImageCollection("COPERNICUS/S2")
.filterDate('2020-01-01','2020-01-30')
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
.filterBounds(geometry)
.median();
var visParamsTrue={bands: ['B4','B3','B2'], min: 0, max: 5000,gamma:1.1};
Map.addLayer(S2_collection.clip(geometry),visParamsTrue,'sentinel-2020');
Map.centerObject(geometry,8);
//Create Training Data
var training=banana.merge(paddy).merge(turmeric).merge(others);
print(training);
var label='landcover'
var bands=['B2','B3','B4','B8'];
var input=S2_collection.select(bands);

var trainImage=input.sampleRegions({
  collection: training,
  properties: [label],
  scale:30
});
 var trainingData=trainImage.randomColumn();
 var trainSet=trainingData.filter(ee.Filter.lessThan('random',0.8));
 var testSet=trainingData.filter(ee.Filter.greaterThanOrEquals('random',0.8));
 
 var classifier=ee.Classifier.smileCart().train(trainSet,label,bands);
 var classifier1=ee.Classifier.smileNaiveBayes().train(trainSet,label,bands);
 var classifier2=ee.Classifier.libsvm().train(trainSet,label,bands);
  var classifier3=ee.Classifier.smileRandomForest(10).train(trainSet,label,bands);
 //var classifier3=ee.Classifier.smileRandomForest().train(trainSet,label,bands);
 var classified=input.classify(classifier);
 var classified1=input.classify(classifier1);
 var classified2=input.classify(classifier2);
 var classified3=input.classify(classifier3);
 
 
 var landcoverpallete=[
   '253494',//water (0)#FF0000','#00FF00','#0000FF','#F5DEB3
   '006837',//forest(1)
   '#FFFF00',//barren(2)
   '#FF00FF',//urban(3)
   ];
   Map.addLayer(classified.clip(geometry),{palette:landcoverpallete, min:0,max:4},'Classification CART');
   Map.addLayer(classified1.clip(geometry),{palette:landcoverpallete, min:0,max:4},'NaiveBayes');
   Map.addLayer(classified2.clip(geometry),{palette:landcoverpallete, min:0,max:4},'SVM');
   // accuracy assessment
   
   // Function to calculate precision, recall, and F1-score
function calculateMetrics(confusionMatrix) {
  var precision = confusionMatrix.getInfo()[2][2] / (confusionMatrix.getInfo()[2][2] + confusionMatrix.getInfo()[0][2]);
  var recall = confusionMatrix.getInfo()[2][2] / (confusionMatrix.getInfo()[2][2] + confusionMatrix.getInfo()[2][0]);
  var f1Score = 2 * (precision * recall) / (precision + recall);
  return {
    precision: precision,
    recall: recall,
    f1Score: f1Score
  };
}

// Accuracy assessment for CART classifier
var confusionMatrix = ee.ConfusionMatrix(
  testSet.classify(classifier)
    .errorMatrix({
      actual: 'landcover',
      predicted: 'classification'
    })
);

var metrics = calculateMetrics(confusionMatrix);

print('Confusion matrix CART:', confusionMatrix);
print('Overall Accuracy CART:', confusionMatrix.accuracy().getInfo());
print('Precision CART:', metrics.precision);
print('Recall CART:', metrics.recall);
print('F1 Score CART:', metrics.f1Score);

// Accuracy assessment for NaiveBayes classifier
var confusionMatrix1 = ee.ConfusionMatrix(
  testSet.classify(classifier1)
    .errorMatrix({
      actual: 'landcover',
      predicted: 'classification'
    })
);

var metrics1 = calculateMetrics(confusionMatrix1);

print('Confusion matrix NaiveBayes:', confusionMatrix1);
print('Overall Accuracy NaiveBayes:', confusionMatrix1.accuracy().getInfo());
print('Precision NaiveBayes:', metrics1.precision);
print('Recall NaiveBayes:', metrics1.recall);
print('F1 Score NaiveBayes:', metrics1.f1Score);

// Accuracy assessment for SVM classifier
var confusionMatrix2 = ee.ConfusionMatrix(
  testSet.classify(classifier2)
    .errorMatrix({
      actual: 'landcover',
      predicted: 'classification'
    })
);

var metrics2 = calculateMetrics(confusionMatrix2);

print('Confusion matrix SVM:', confusionMatrix2);
print('Overall Accuracy SVM:', confusionMatrix2.accuracy().getInfo());
print('Precision SVM:', metrics2.precision);
print('Recall SVM:', metrics2.recall);
print('F1 Score SVM:', metrics2.f1Score);

// Accuracy assessment for Random Forest classifier
var confusionMatrix3 = ee.ConfusionMatrix(
  testSet.classify(classifier3)
    .errorMatrix({
      actual: 'landcover',
      predicted: 'classification'
    })
);

var metrics3 = calculateMetrics(confusionMatrix3);

print('Confusion matrix RF:', confusionMatrix3);
print('Overall Accuracy RF:', confusionMatrix3.accuracy().getInfo());
print('Precision RF:', metrics3.precision);
print('Recall RF:', metrics3.recall);
print('F1 Score RF:', metrics3.f1Score);

  var confusionMatrix=ee.ConfusionMatrix(testSet.classify(classifier)
  .errorMatrix({
    actual:'landcover',
    predicted:'classification'
     
  }));
  var confusionMatrix1=ee.ConfusionMatrix(testSet.classify(classifier1)
  .errorMatrix({
    actual:'landcover',
    predicted:'classification'
     
  }));
  var confusionMatrix2=ee.ConfusionMatrix(testSet.classify(classifier2)
  .errorMatrix({
    actual:'landcover',
    predicted:'classification'
     
  }));
    var confusionMatrix3=ee.ConfusionMatrix(testSet.classify(classifier3)
  .errorMatrix({
    actual:'landcover',
    predicted:'classification'
      
  }));
  
