function [X_train,y_train,X_test,y_test] = loadDataset(path)
    rng(20);
    dataset = readtable(path,'VariableNamingRule','preserve');
    indexes = cvpartition(size(dataset,1),'HoldOut',0.2).test;
    dataTrain = dataset(~indexes,:);
    dataTest = dataset(indexes,:);
    
    dataset_train = dataTrain(:,1:8);
    dataset_train = dataset_train{:,:};
    dataset_test = dataTest(:,1:8);
    dataset_test = dataset_test{:,:};
    
    dataset_train(any(isnan(dataset_train),2),:) = []; 
    dataset_test(any(isnan(dataset_test),2),:) = []; 
    
    
    X_train = dataset_train(:,2:8);
    y_train = dataset_train(:,1);
    X_test = dataset_test(:,2:8);
    y_test = dataset_test(:,1);
    
    
    X_train = normalize(X_train);
    y_train = normalize(y_train);
    X_test = normalize(X_test);
    y_test = normalize(y_test);
end
