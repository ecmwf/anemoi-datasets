x, y = ds[n : n + 1]
y_hat = model.predict(x)
loss = model.loss(y, y_hat)
