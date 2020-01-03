function p=logclassify(weight,bias,xTest);
%% classifier process
hTest = weight'*xTest+bias; 
gTest= 1.0 ./ (1.0 + exp(-hTest)); 
p =double( (gTest >= 0.5));
end
