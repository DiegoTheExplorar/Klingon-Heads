# train model
model.fit(
    [english_train_padded, klingon_train_input], klingon_train_target,
    batch_size=64, epochs=20, validation_data=([english_test_padded, klingon_test_input], klingon_test_target)
)

# Evaluate the model on test data
test_loss = model.evaluate([english_test_padded, klingon_test_input], klingon_test_target)
print(f'Test Loss: {test_loss}')
