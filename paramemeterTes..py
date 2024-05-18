
for batch_size in [32, 64, 128]:
    for epochs in [20, 50, 100]:
        print(f'Training with batch size: {batch_size} and epochs: {epochs}')
        history = model.fit(
            [english_train_padded, klingon_train_input], klingon_train_target,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([english_test_padded, klingon_test_input], klingon_test_target),
            callbacks=[early_stopping, reduce_lr]
        )
        # Evaluate the model
        test_loss, test_metric = model.evaluate([english_test_padded, klingon_test_input], klingon_test_target)
        print(f'Test Loss: {test_loss}, Test Metric: {test_metric}')
