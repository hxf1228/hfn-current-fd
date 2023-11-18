function features = featurecacul(input)
feature1 = skewness(input);
feature2 = kurtosis(input);
feature3 = peak2rms(input);
feature4 = shapefactor(input);
feature5 = impulsefactor(input);
feature6 = clearancefactor(input);
features=[feature1 feature2 feature3 feature4 feature5 feature6];
end


function shapefactor=shapefactor(a)
%求信号的波形因子
shapefactor=rms(a)./mean(abs(a));
end
function impulsefactor=impulsefactor(a)
%求信号的脉冲因子
impulsefactor=peak(a)./mean(abs(a));
end

function pk=peak(a)
%求信号的峰值
pk=max(a)-min(a);
end

function clearancefactor=clearancefactor(a)
%求信号的裕度因子
clearancefactor=peak(a)./mean(sqrt(abs(a))).^2;
end
