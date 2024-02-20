import torch
import time
# Create a batched tensor of size (batch_size, 100, 100)
batch_size = 1000
tensor = torch.randn(batch_size, 50, 50)

def run_slice(tensor):
    torch.sum(tensor)
    torch.mean(tensor)
    torch.std(tensor)
    return torch.var(tensor)

# Iterate over the batch dimension using a for loop


start_time = time.time()


for i in range(batch_size):
    
    run_slice(tensor[i])
    # Do something with the batch

end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Elapsed time for iterating over shape: {elapsed_time} seconds")

start_time = time.time()

# Iterate over the batch dimension using torch.split()
batch_list = torch.split(tensor, 1, dim=0)
for batch in batch_list:
    run_slice(batch)
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Elapsed time for iterating over splits: {elapsed_time} seconds")

start_time = time.time()
# Iterate over the batch dimension using torch.chunk()
batch_list = torch.chunk(tensor, batch_size, dim=0)
print(batch_list[0].shape)
for batch in batch_list:
    # Do something with the batch
    run_slice(batch)
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Elapsed time for iterating over chunks: {elapsed_time} seconds")

start_time = time.time()
batch_list = torch.unbind(tensor, dim=0)
for batch in batch_list:
    # Do something with the batch
    run_slice(batch)
end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Elapsed time for iterating over torch.bind: {elapsed_time} seconds")

start_time = time.time()
result=torch.stack([run_slice(batch) for batch in batch_list],dim=0)

end_time = time.time() 
elapsed_time = end_time - start_time 
print(f"Elapsed time for inline: {elapsed_time} seconds")

start_time = time.time()