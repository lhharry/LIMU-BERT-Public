% data 还是正常转(假设全是数值)
data_arr  = table2array(data);
data_cols = data.Properties.VariableNames;

% labels 拆开处理
label_header = labels.Header;              % 数值列 → double 向量
label_text   = labels.Label;               % cell 列 → cell array of char

% 如果 Label 里是字符串想转成数字编码(便于训练):
[label_id, label_names] = grp2idx(label_text);
% label_id    : 3045x1 数值(1,2,3...)
% label_names : 每个编号对应的原始字符串

label_cols = labels.Properties.VariableNames;

save('converted.mat', ...
     'data_arr', 'data_cols', ...
     'label_header', 'label_text', 'label_id', 'label_names', 'label_cols', ...
     'stairHeight', 'subject', 'transLegAscent', 'transLegDescent', ...
     'trialEnds', 'trialStarts', '-v7');