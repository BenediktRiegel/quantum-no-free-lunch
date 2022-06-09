def generate_config(learning_rate, batch_size, num_epochs, shuffle):
    config = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle = shuffle)
    return config