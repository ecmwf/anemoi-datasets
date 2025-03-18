input = open_dataset(zip=[low_res_dataset, high_res_orography_dataset])
output = open_dataset(high_res_dataset)

ds = open_dataset(x=input, y=output)

for (x, orography), y in ds:
    y_hat = model(x, orography)
    loss = criterion(y_hat, y)
    loss.backward()
