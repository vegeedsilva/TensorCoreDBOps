import matplotlib.pyplot as plt


#Join line graph
join_tensor = [0.072768, 0.073632, 0.080832, 0.349824, 1.870848, 12.81168, 111.052094, 1045.70568]
join_chunking_tensor = [0.072768, 0.073632, 0.080832, 0.224960, 0.394368, 0.802944, 1.265312, 2.553824]
join_without_tensor = [0.093856, 0.107744, 0.2784, 1.422464, 11.312032, 84.610847, 610.050964, 4571.69]
join_without_chunking_tensor = [0.093856, 0.107744, 0.2784, 0.556896, 1.042016, 1.882912, 3.843488, 7.581408]
join_gpu = [0.014464, 0.003392, 0.0035, 0.003488, 0.003456, 0.003488, 0.003520, 0.003648]

#x = ['16', '256', '1024', '2048', '4096', '8192', '16384', '32768']
x = ['2^4', '2^8', '2^10', '2^11', '2^12']
plt.plot(x, join_tensor[:-3], 'r--', x, join_chunking_tensor[:-3], 'g--', x, join_without_tensor[:-3], 'b--', x, join_without_chunking_tensor[:-3], 'y--', x, join_gpu[:-3], 'c--')
plt.legend(['Join operation using tensors', 'Join operation using tensors with chunking', 'Join operation without tensors', 'Join operation without tensors and chunking', 'Join operation in GPU' ])
plt.xlabel('Input Table sizes')
plt.ylabel('Time taken for execution (ms)')
plt.title('Execution times of various join operation approaches')
plt.show()

#Join operation tensorcore bar graph
join_mat_mul = [0.059744, 0.071040, 0.077696, 0.34668, 1.867840, 12.808288, 111.048767]
join_bit_flip = [0.013024, 0.00259, 0.003136, 0.003136, 0.003008, 0.003392, 0.003328]
x = ['2^4', '2^8', '2^10', '2^11', '2^12', '2^13', '2^14']
width = 0.4
p1 = plt.bar(x[:4], join_mat_mul[:4], width)
p2 = plt.bar(x[:4], join_bit_flip[:4], width, bottom=join_mat_mul[:4])
plt.ylabel('Total Time for execution')
plt.title('Join operation execution times in Tensor Cores')
plt.legend((p1[0], p2[0]), ('Matrix multiplication time', 'Bit Flipping time'))
plt.show()

#SELECT line graph
select_tensor = [0.153504, 0.142240, 0.101408, 0.311264, 1.816416, 12.728608, 108.859680, 1047.962036]
select_chunking_tensor = [0.153504, 0.142240, 0.101408,  0.118304, 0.227616, 0.449376, 0.826304,	1.610016]
select_without_tensor = [0.213568,0.179232,0.301600,1.427936,10.999969,84.326561,572.089966,4558.312988]
select_without_chunking_tensor = [0.213568,0.179232,0.301600, 0.556896, 1.042016, 1.882912, 3.843488, 7.581408]
select_gpu = [0.075072,	 0.003616,	0.003648,	0.003360,	0.003392,	0.003392,	0.003360,	0.003392]

#x = ['16', '256', '1024', '2048', '4096', '8192', '16384', '32768']
x = ['2^4', '2^8', '2^10', '2^11', '2^12']

plt.plot(x, select_tensor[:-3], 'r--', x, select_chunking_tensor[:-3], 'g--', x, select_without_tensor[:-3], 'b--', x, select_without_chunking_tensor[:-3], 'y--', x, select_gpu[:-3], 'c--')
plt.legend(['Select where operation using tensors', 'Select where operation using tensors with chunking', 'Select where operation without tensors', 'Select where operation without tensors and chunking', 'Select where operation in GPU' ])
plt.xlabel('Input Table sizes')
plt.ylabel('Time taken for execution (ms)')
plt.title('Execution times of various Select where operation approaches')
plt.show()


#select operation tensorcore bar graph
select_mat_mul = [0.123936, 0.139392, 0.098304, 0.308256]
select_bit_flip = [0.02, 0.002848, 0.003104, 0.003008]
x = ['2^4', '2^8', '2^10', '2^11', '2^12', '2^13', '2^14']
width = 0.4
p1 = plt.bar(x[:4], select_mat_mul[:4], width)
p2 = plt.bar(x[:4], select_bit_flip[:4], width, bottom=select_mat_mul[:4])
plt.ylabel('Total Time for execution')
plt.title('select operation execution times in Tensor Cores')
plt.legend((p1[0], p2[0]), ('Matrix multiplication time', 'Bit Flipping time'))
plt.show()

