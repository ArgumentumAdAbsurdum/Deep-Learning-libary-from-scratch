import torch
import torch.nn as nn
import pandas as pd
import time


df = pd.read_csv("datasets/mnist_train.csv", header = None)


x = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
x = x / 255.0

train_size = int(0.8 * len(x))

train_x = x[:train_size]
train_y = y[:train_size]

test_x = x[train_size:]
test_y = y[train_size:]



def benchmark(batch_size, epochs):

    model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    )

    sgd = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    start = time.time()
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i : i + batch_size]
            batch_y = train_y[i : i + batch_size]

            sgd.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            sgd.step()

    end = time.time()
    print(f"[batch_size = {batch_size}, epochs = {epochs} => time :  {(end - start):.2f}s, ", end="")

    with torch.no_grad():
        output = model(test_x)
        predicted = torch.argmax(output, dim = 1)
        accuracy = (predicted == test_y).float().mean()
        print(f"accuracy : {accuracy:.2f}]")





def benchmark_adam(batch_size, epochs):

    model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    )

    adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i : i + batch_size]
            batch_y = train_y[i : i + batch_size]

            adam.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            adam.step()

    end = time.time()
    print(f"[batch_size = {batch_size}, epochs = {epochs} => time :  {(end - start):.2f}s, ", end="")

    with torch.no_grad():
        output = model(test_x)
        predicted = torch.argmax(output, dim = 1)
        accuracy = (predicted == test_y).float().mean()
        print(f"accuracy : {accuracy:.2f}]")





if __name__ == "__main__":
    print(f"--------------------------------------------------------")
    benchmark(batch_size=1, epochs=1)
    benchmark(batch_size=2, epochs=1)
    benchmark(batch_size=4, epochs=1)
    benchmark(batch_size=8, epochs=1)
    benchmark(batch_size=16, epochs=20)
    benchmark(batch_size=32, epochs=20)
    benchmark(batch_size=64, epochs=20)


    print(f"--------------------------------------------------------")
    benchmark_adam(batch_size=1, epochs=1)
    benchmark_adam(batch_size=2, epochs=1)
    benchmark_adam(batch_size=4, epochs=1)
    benchmark_adam(batch_size=8, epochs=1)
    benchmark_adam(batch_size=16, epochs=20)
    benchmark_adam(batch_size=32, epochs=20)
    benchmark_adam(batch_size=64, epochs=20)

    











