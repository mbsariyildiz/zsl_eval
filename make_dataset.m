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
Xtr_all = [Xtr; Xva];
Xtrva = features(:, trainval_loc)';
Xte_seen = features(:, test_seen_loc)';
Xte_unseen = features(:, test_unseen_loc)';
Xtrva_all = [Xtrva; Xte_seen; Xte_unseen];

% split labels
Ltr = labels(train_loc, :);
Lva = labels(val_loc, :);
Ltr_all = [Ltr; Lva];
Ltrva = labels(trainval_loc, :);
Lte_seen = labels(test_seen_loc, :);
Lte_unseen = labels(test_unseen_loc, :);
Ltrva_all = [Ltrva; Lte_seen; Lte_unseen];

% split attributes
Str = att(:, Ltr)';
Sva = att(:, Lva)';
Str_all = [Str; Sva];
Strva = att(:, Ltrva)';
Ste_seen = att(:, Lte_seen)';
Ste_unseen = att(:, Lte_unseen)';
Strva_all = [Strva; Ste_seen; Ste_unseen];

% split class labels
Ctr = unique(Ltr);
Cva = unique(Lva);
Ctrva = [Ctr; Cva];
Cte_seen = unique(Lte_seen);
Cte_unseen = unique(Lte_unseen);
Call = unique(labels);

% Produce one-hot versions of class labels. One-hot coding scheme 
% is used such that only dataset type of interest is considered,
% i.e. 'tr', 'va', 'trva.
% These labels are meant to be used in training time
% to enable processing batch of samples. They do not represent 
% global class labels, therefore they shouldn't be used in testing.
Ltr_oh = one_hot(Ltr, Ctr);
Lva_oh = one_hot(Lva, Cva);
Ltrva_oh = one_hot(Ltrva, Ctrva);
Lte_seen_oh = one_hot(Lte_seen, Cte_seen);
Lte_unseen_oh = one_hot(Lte_unseen, Cte_unseen);

% split attributes of each class
Str_gt = att(:, Ctr)';
Sva_gt = att(:, Cva)';
Strva_gt = att(:, Ctrva)';
Ste_seen_gt = att(:, Cte_seen)';
Ste_unseen_gt = att(:, Cte_unseen)';
Sall_gt = att';

save(fname, 'Xtr', 'Xva', 'Xtr_all', 'Xtrva', 'Xte_seen', 'Xte_unseen', 'Xtrva_all', ...
            'Str', 'Sva', 'Str_all', 'Strva', 'Ste_seen', 'Ste_unseen', 'Strva_all', ...
            'Ltr', 'Lva', 'Ltr_all', 'Ltrva', 'Lte_seen', 'Lte_unseen', 'Ltrva_all', ...
            'Ltr_oh', 'Lva_oh', 'Ltrva_oh', 'Lte_seen_oh', 'Lte_unseen_oh', ...
            'Ctr', 'Cva', 'Ctrva', 'Cte_seen', 'Cte_unseen', 'Call', ...
            'Str_gt', 'Sva_gt', 'Strva_gt', 'Ste_seen_gt', 'Ste_unseen_gt', 'Sall_gt', ...
    '-v7.3');

end

function [L_oh] = one_hot(L, C)
L_oh = zeros(size(L, 1), length(C));
for c = 1:length(C)
    class_id = C(c);
    inds = find(L == class_id);
    L_oh(inds, c) = 1;
end
end