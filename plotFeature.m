function plotFeature(feature, mask)
  labels = unique(mask)
  colors = 'rbgy';
  hold on
  for i=labels'
    data = feature(find(mask==i))
    plot(data,['*' colors(i)]);
  end
  xlabel('index');
  ylabel('value');
  hold off
end
