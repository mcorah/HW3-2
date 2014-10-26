function plotFeature(feature, data)
  labels = unique(data.Label)
  colors = 'rbgy';
  column = data{:,feature}
  figure();
  hold on
  for i=labels'
    plot_data = column(find(data.Label==i))
    plot(plot_data,['*' colors(i)]);
  end
  xlabel('index');
  ylabel('value');
  title(feature)
  hold off
end
