import torch


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1] #2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) #dot product

print(attn_scores_2)
#tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

#normalization 
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
#attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
#Sum: tensor(1.0000)

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0) #cause we are summing all the entries in the row 
attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum()) 
#could lead to overflow problems 

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("sum", attn_weights_2.sum())
# they will result same for now since these values dont have huge variations, but with big parameters its good to use softmax method and it gives the postive(+) solutions

context_vec_2 = torch.zeros(query.shape) # start with empty vector of zeros(0,0,0)
for i, x_i in enumerate(inputs):#i is the words position, and x_i is the words vector like 0.43, 0.15, 0.89 for your, multiply the weight with each vector of the word
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
#tensor([0.4419, 0.6515, 0.5683])

attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)

print(attn_scores)
#tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#[0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#[0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#[0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#[0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

#for loops are slow and we can achieve the same results using mutliplicaton

attn_scores = inputs @ inputs.T
print(attn_scores)
#tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#[0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#[0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#[0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#[0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])


#normalization
attn_weights = torch.softmax(attn_scores, dim =-1)
print(attn_weights)
#dim -1 means we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.
#if the attn_scores is 2D( rows and columns ), dim = -1 will normalize across the columns so that the values in each row(summing over the column dimension) sum up to 1
#dim=-1 applies softmax along the last dimension. For a 2D tensor, that means each row gets normalized, so every row adds up to 1.
#Why this is what we want: each row is one word's scores against all the other words. Row 2 is "journey" vs every word. Softmax per row turns that into "how much 'journey' attends to each word", summing to 100%.

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
#tensor([[0.4421, 0.5931, 0.5790],
        #[0.4419, 0.6515, 0.5683],
        #[0.4431, 0.6496, 0.5671],
        #[0.4304, 0.6298, 0.5510],
        #[0.4671, 0.5910, 0.5266],
        #[0.4177, 0.6503, 0.5645]])