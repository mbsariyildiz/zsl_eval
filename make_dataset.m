function [] = make_dataset( dataset, fname)
%CREATE_DATASET 
%   Splits fields of specified "dataset" and stores each field into the 
%   file "fname" which is compatible with hdf5 so that it can directly be
%   loaded into numpy by h5py. You can use this function to modify .mat files provided by
%   Yongqin Xian. 
%   
%   INPUTS:
%       dataset,        string indicating name of dataset folder, 
%                       e.g. one of ['CUB', 'AWA', 'SUN', 'APY']. 
%       fname,          path indicating where to save resulting fields

load(['../data/' dataset '/att_splits.mat']);
load(['../data/' dataset '/res101.mat'], 'features', 'labels');

% split resnet featuers
Xtr = features(:, train_loc)';
Xva = features(:, val_loc)';
Xtrva = features(:, trainval_loc)';
Xte_seen = features(:, test_seen_loc)';
Xte_unseen = features(:, test_unseen_loc)';

% split labels
Ltr = labels(train_loc, :);
Lva = labels(val_loc, :);
Ltrva = labels(trainval_loc, :);
Lte_seen = labels(test_seen_loc, :);
Lte_unseen = labels(test_unseen_loc, :);

% split attributes
Str = att(:, Ltr)';
Sva = att(:, Lva)';
Strva = att(:, Ltrva)';
Ste_seen = att(:, Lte_seen)';
Ste_unseen = att(:, Lte_unseen)';

% split class labels
Ctr = unique(Ltr);
Cva = unique(Lva);
Ctrva = [Ctr; Cva];
Cte_seen = unique(Lte_seen);
Cte_unseen = unique(Lte_unseen);
Call = unique(labels);

% split attributes of each class
Str_gt = att(:, Ctr)';
Sva_gt = att(:, Cva)';
Strva_gt = att(:, Ctrva)';
Ste_seen_gt = att(:, Cte_seen)';
Ste_unseen_gt = att(:, Cte_unseen)';
Sall_gt = att';

save(fname, 'Xtr', 'Xva', 'Xtrva', 'Xte_seen', 'Xte_unseen', ...
            'Str', 'Sva', 'Strva', 'Ste_seen', 'Ste_unseen', ...
            'Ltr', 'Lva', 'Ltrva', 'Lte_seen', 'Lte_unseen', ...
            'Ctr', 'Cva', 'Ctrva', 'Cte_seen', 'Cte_unseen', 'Call', ...
            'Str_gt', 'Sva_gt', 'Strva_gt', 'Ste_seen_gt', 'Ste_unseen_gt', 'Sall_gt', ...
    '-v7.3');

end

