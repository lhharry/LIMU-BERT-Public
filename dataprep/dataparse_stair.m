% data 还是正常转(假设全是数值)
data_arr  = table2array(data);
data_cols = data.Properties.VariableNames;

% speed 拆开处理
label_header = speed.Header;               % 数值列 → double 向量
speed_vals   = speed.Speed;                % 数值列 → double 向量

% 根据 speed 生成 label_id：speed==0 → 1，其它 → 8
label_id = ones(size(speed_vals));         % 先全部设为 1
label_id(speed_vals ~= 0) = 8;             % 非零的改成 8

% 为了和原来结构保持一致，也生成对应的文本标签和名称表
label_names = {'speed_zero'; 'speed_nonzero'};   % 1 -> 'speed_zero', 8 -> 'speed_nonzero'
label_text  = repmat({'speed_zero'}, size(speed_vals));
label_text(speed_vals ~= 0) = {'speed_nonzero'};

label_cols = speed.Properties.VariableNames;

save('converted.mat', ...
    'data_arr', 'data_cols', ...
    'label_header', 'speed_vals', 'label_text', 'label_id', 'label_names', 'label_cols', ...
    'trialEnds', 'trialStarts', '-v7');