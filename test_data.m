data = readtable('postureData.txt');

labels = {'triumphant' 'concentrated' 'defeated' 'frustrated'};
plotFeature(data.XpositionHip,data.Label);
