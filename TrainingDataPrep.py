# Prepare target data for training
klingon_train_input = klingon_train_padded[:, :-1] # The decoder input, which is the Klingon sentence shifted by one position to the right for training data.
klingon_train_target = klingon_train_padded[:, 1:] # The target output, which is the same sentence shifted by one position to the left for training data.
klingon_train_target = np.expand_dims(klingon_train_target, -1)

# Prepare target data for testing
klingon_test_input = klingon_test_padded[:, :-1] # The decoder input for testing data.
klingon_test_target = klingon_test_padded[:, 1:] # The target output for testing data.
klingon_test_target = np.expand_dims(klingon_test_target, -1)
