for x, y in ds:
    y_hat = model(x)
    loss = criterion(y_hat, y)
    loss.backward()
