data = csvread('postureData.csv',1,0);

labels = {'triumphant' 'concentrated' 'defeated' 'frustrated'};
plotFeature(data.XpositionHip,data.Label);