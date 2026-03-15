import torch
import torch.nn as nn
import pandas as pd
import time


train_df = pd.read_csv("datasets/mnist_train.csv", header = None)
test_df = pd.read_csv("datasets/mnist_test.csv", header = None)

train_x = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32)
train_y = torch.tensor(train_df.iloc[:, 0].values, dtype=torch.long)

test_x = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32)
test_y = torch.tensor(test_df.iloc[:, 0].values, dtype=torch.long)

train_x = train_x / 255.0
test_x  = test_x / 255.0

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

sgd = torch.optim.SGD(model.parameters(), lr=0.001)
adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

loss_fn = nn.CrossEntropyLoss()



def benchmark(batch_size, epochs, optimizer):

    start = time.time()
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i : i + batch_size]
            batch_y = train_y[i : i + batch_size]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    end = time.time()
    print(f"[batch_size = {batch_size}, epochs = {epochs} => time :  {(end - start):.2f}s, ", end="")

    with torch.no_grad():
        output = model(test_x)
        predicted = torch.argmax(output, dim = 1)
        accuracy = (predicted == test_y).float().mean()
        print(f"accuracy : {accuracy:.2f}]")




if __name__ == "__main__":
    print(f"--------------------------------------------------------")
    benchmark(batch_size=1, epochs=1, optimizer=sgd)
    benchmark(batch_size=2, epochs=1, optimizer=sgd)
    benchmark(batch_size=4, epochs=1, optimizer=sgd)
    benchmark(batch_size=8, epochs=1, optimizer=sgd)
    benchmark(batch_size=16, epochs=1, optimizer=sgd)
    benchmark(batch_size=32, epochs=1, optimizer=sgd)
    benchmark(batch_size=64, epochs=1, optimizer=sgd)


    print(f"--------------------------------------------------------")
    benchmark(batch_size=1, epochs=1, optimizer=adam)
    benchmark(batch_size=2, epochs=1, optimizer=adam)
    benchmark(batch_size=4, epochs=1, optimizer=adam)
    benchmark(batch_size=8, epochs=1, optimizer=adam)
    benchmark(batch_size=16, epochs=1, optimizer=adam)
    benchmark(batch_size=32, epochs=1, optimizer=adam)
    benchmark(batch_size=64, epochs=1, optimizer=adam)

    











